import torch
import rdkit
from numpy.core.fromnumeric import argmax
import os
from torch_geometric.utils import to_networkx
from torch_geometric.data import Data
from tqdm import tqdm
from RawZINC import RawZINC

bonds_dec = {1: '-', 2: '=', 3: '#', 4: '~'}

zinc_bond_dict = {'NONE':0, 'SINGLE': 1, 'DOUBLE': 2, 'TRIPLE': 3}
zinc_bond_dec = {0:' ', 1: '-', 2: '=', 3: '#'}
zinc_atom_dict = {'C': 0, 'O': 1, 'N': 2, 'F': 3, 'C H1': 4, 'S': 5, 'Cl': 6, 'O -': 7, 'N H1 +': 8, 'Br': 9,
                  'N H3 +': 10, 'N H2 +': 11, 'N +': 12, 'N -': 13, 'S -': 14, 'I': 15, 'P': 16, 'O H1 +': 17,
                  'N H1 -': 18, 'O +': 19, 'S +': 20, 'P H1': 21, 'P H2': 22, 'C H2 -': 23, 'P +': 24, 'S H1 +': 25,
                  'C H1 -': 26, 'P H1 +': 27}

def walks_from_one_node(start_nodeIdx, nodes_sym, adj_list, walk_length):
    queue = []
    visited = []
    kpaths = []  # depth=k
    allpaths = []
    path_str_syb = []
    kpaths_syb = []
    allpaths_syb = []

    queue.append(start_nodeIdx)
    visited.append(start_nodeIdx)
    path_str = []
    deep = 0
    k = walk_length

    while len(queue) != 0:
        if deep <= k:
            centerIdx = queue.pop(0)
            if len(path_str) != 0:
                idx = int(path_str[-1])
                nbs_id = adj_list[idx].keys()
                if centerIdx in nbs_id:
                    if len(queue) != 0:
                        bond = queue.pop(0)
                        path_str.append(bond)
                        path_str_syb.append(bond)
                    path_str.append(str(centerIdx))
                    path_str_syb.append(nodes_sym[centerIdx])
                    allpaths.append("".join(path_str))
                    allpaths_syb.append("".join(path_str_syb))
                else:
                    if len(queue) != 0:
                        bond = queue.pop(0)
                        continue
            else:
                if len(queue) != 0:
                    bond = queue.pop(0)
                    path_str.append(bond)
                    path_str_syb.append(bond)
                path_str.append(str(centerIdx))
                path_str_syb.append(nodes_sym[centerIdx])
                allpaths.append("".join(path_str))
                allpaths_syb.append("".join(path_str_syb))

            deep = deep + 1
            if deep == k:
                kpaths.append("".join(path_str))
                kpaths_syb.append("".join(path_str_syb))
            if centerIdx not in visited:
                visited.append(centerIdx)

            centerIdx_flag = 0  # - 当前中心点的可走邻居数: =0则没有新节点进栈, =n则有n个节点入栈
            for neibIdx in adj_list[centerIdx].keys():
                if (neibIdx not in visited) and (deep < k):
                    # bondtype_onehot = adj_list[centerIdx][neibIdx]['edge_attr']
                    # bondtype_sym = bonds_dec[argmax(bondtype_onehot)]  # - 取出与该邻居节点之间的BondType# 0-single,1-double,2-triple,3-aromatic

                    bondtype = adj_list[centerIdx][neibIdx]['edge_attr']
                    bondtype_sym = bonds_dec[bondtype]
                    queue.append(neibIdx)  # - 邻居节点id先入栈
                    queue.append(bondtype_sym)  # - 然后再将BondType入栈
                    centerIdx_flag = centerIdx_flag + 1  # - 记录在center-node的邻居中，新入栈的邻居节点的个数

            if deep == k or centerIdx_flag == 0:  # - 没有新节点进栈，或路径长度==k
                if len(path_str) == 0:
                    print("path_str==0  ||||||  ", start_nodeIdx)
                if len(path_str) > 1:
                    path_str.pop()  # nodeidx
                    path_str.pop()  # bond
                    path_str_syb.pop()  # - 先将nb-id出栈
                    path_str_syb.pop()  # - 再将与nb-id相连的BondType出栈
                    deep = deep - 1  # - 路径长度 减1

                else:
                    if len(path_str) == 1:
                        path_str.pop()  # nodeidx
                        path_str_syb.pop()  # - 先将nb-id出栈
                        deep = deep - 1  # - 路径长度 减1

    return allpaths, allpaths_syb, kpaths, kpaths_syb

def getMetapathPairs(start_nodesIdx, nodes_sym, metapath_copus, adj_list, walk_length):
    watchAllPaths = []
    watchAllPathsSymbol = []
    totalpaths = []
    totalpaths_syb = []
    mp_us = []
    mp_vs = []
    find_mp = []
    find_mp_symbol = []
    for startIdx in start_nodesIdx:
        paths, paths_syb, kpaths, kpaths_syb = walks_from_one_node(start_nodeIdx=startIdx,
                                                                   nodes_sym=nodes_sym,
                                                                   adj_list=adj_list,
                                                                   walk_length=walk_length)
        watchAllPaths.extend(paths)
        watchAllPathsSymbol.extend(paths_syb)
        for i, path in enumerate(paths):
            reversed_path = path[::-1]
            if path not in totalpaths:  # + and (int(path[0]) != int(path[-1])): #~ 去除重复和自环
                if (int(path[0]) != int(path[-1])) or (int(path[0]) == int(path[-1]) and (len(path) > 1)):
                    totalpaths.append(path)
                    totalpaths_syb.append(paths_syb[i])
                    if paths_syb[i] in metapath_copus:
                        find_mp.append(path)
                        find_mp_symbol.append(paths_syb[i])
                        mp_us.append(int(path[0]))
                        mp_vs.append(int(path[-1]))

    return mp_us, mp_vs, find_mp, find_mp_symbol

def label2onehot(labels, dim):  # ~ Convert label indices to one-hot vectors.
    out = torch.zeros(list(labels.size()) + [dim])
    out.scatter_(len(out.size()) - 1, labels.unsqueeze(-1), 1.)
    return out

def get_Iterable_adj_dict(adj_list):
    g_adj_list = {}
    for a in adj_list:
        center_nodeIdx, neighbors_dict = a
        g_adj_list[center_nodeIdx] = neighbors_dict  # - construct an iterable adj-list
    return g_adj_list


def uniqueMPSubgraphs(mp_subgs, mp_subgs_sym):
    lst = mp_subgs
    mp_subgs_sym
    for i, ele in enumerate(lst):
        rev = ele[::-1]
        cnt = lst.count(rev)
        while (rev in lst) and (lst.count(rev) > 0):
            idx = lst.index(rev)
            lst.remove(rev)
            mp_subgs_sym.pop(idx)
    return lst, mp_subgs_sym

def load_MP_Dataset(dataset_dir, split, save_dir):
    zinc = RawZINC(dataset_dir, subset=False, split=split)
    zinc.process()
    # data = zinc.data  # ~ Data(edge_attr=[4883516, 4], edge_index=[2, 4883516], idx=[130831], name=[130831], pos=[2359210, 3], x=[2359210, 11], y=[130831, 19], z=[2359210])
    # num_classes = zinc.num_classes  # 19
    # nodes_symbols = {'H': 1, 'C': 6, 'N': 7, 'O': 8, 'F': 9, 'P':15, 'S':16, 'I':53, 'Cl':17, "Br":35}
    nodes_types_enc = {'C': 0, 'O': 1, 'N': 2, 'F': 3, 'C H1': 4, 'S': 5, 'Cl': 6, 'O -': 7, 'N H1 +': 8, 'Br': 9,
                   'N H3 +': 10, 'N H2 +': 11, 'N +': 12, 'N -': 13, 'S -': 14, 'I': 15, 'P': 16, 'O H1 +': 17,
                   'N H1 -': 18, 'O +': 19, 'S +': 20, 'P H1': 21, 'P H2': 22, 'C H2 -': 23, 'P +': 24, 'S H1 +': 25,
                   'C H1 -': 26, 'P H1 +': 27}
    nodes_types_dec = {nodes_types_enc[key]:key for key in nodes_types_enc.keys()}

    bonds_types = {rdkit.Chem.rdchem.BondType.SINGLE: 1,
                   rdkit.Chem.rdchem.BondType.DOUBLE: 2,
                   rdkit.Chem.rdchem.BondType.TRIPLE: 3}
    bonds_dec = {bonds_types[key]: key for key in bonds_types.keys()}
    bonds_dec[4] = 'mp-edge'

    # nodes_symbols_dec = {nodes_symbols[key]: key for key in nodes_symbols.keys()}
    pyg_list = []
    metapath_copus = [
        "C=C-C-O", "O-C-C=C",
        "C#C-C-O", "O-C-C#C",
        "C-C=O", "O=C-C",
        "O-C=O", "O=C-O",
        "C-O-C=O", "O=C-O-C",
        "C#C-C", "C-C#C",
        "N#C-C", "C-C#N",
        "O-C-N", "N-C-O",
        "N-C=O", "O=C-N",
        "C-O-C",
        "O-N-C", "C-N-O",
        "C-O-O-C",
        "C-O-N-C", "C-N-O-C",
        "C-O-N-F", "F-N-O-C",
        "S-C=C-C", "S-C=C-C",
        'C-C H1-C=O', 'O=C-C H1-C',
        'N-C H1-C=O', "O=C-C H1-N"
    ]

    for i, g in enumerate(tqdm(zinc, desc="mp-processing")):
        g2nx = to_networkx(data=g, edge_attrs=['edge_attr'], to_undirected=True)
        adj_list = g2nx.adjacency()  # - the returned adj-list is disiterable
        g_adj_list = get_Iterable_adj_dict(adj_list=adj_list)

        # =#######################     obtain adj-matrix and adj-list     ################################
        # print("########     obtain Meta-Path Neighbors    ######")
        start_nodesIdx = []
        g2nx.x = g.x
        g2nx.y = g.y
        g2nx.edge_index = g.edge_index
        g2nx.edge_attr = g.edge_attr
        x = g.x.numpy().tolist().copy()
        g2nx.nodes_sym = [nodes_types_dec[idx[0]] for idx in x]
        for i, symbol in enumerate(g2nx.nodes_sym):
            start_nodesIdx.append(i)
        walk_length = 4
        if len(g2nx.nodes_sym) < walk_length:
            walk_length = len(g2nx.nodes_sym)
        mp_us, mp_vs, find_mp, find_mp_symbol = getMetapathPairs(start_nodesIdx=start_nodesIdx,
                                                                 nodes_sym=g2nx.nodes_sym,
                                                                 metapath_copus=metapath_copus,
                                                                 adj_list=g_adj_list,
                                                                 walk_length=walk_length)
        uni_subgs, uni_subgs_sym = uniqueMPSubgraphs(find_mp, find_mp_symbol)
        mp_us = []
        mp_vs = []
        for ele in uni_subgs:
            mp_us.append(int(ele[0]))
            mp_vs.append(int(ele[-1]))
        # =#######################   copy graph-info & adding the fake paths based on metapaths     ################################
        edge_attr = g2nx.edge_attr  # one-hot edge type
        edge_index = g2nx.edge_index

        J_edges_attr = edge_attr.numpy().tolist().copy()

        J_edges_attr = label2onehot(labels=torch.tensor(J_edges_attr).to(torch.int64), dim=4)

        J_edges_type = {i: argmax(e_attr) for i, e_attr in enumerate(J_edges_attr)}
        J_us = edge_index[0].numpy().tolist().copy()
        J_vs = edge_index[1].numpy().tolist().copy()
        J_nodes_sym = g2nx.nodes_sym

        # . J_edge_index : tensor([[0, 0, 0, 0, 1, 1, 1, 2, 3, 4, 5, 6, 0, 4], [1, 3, 4, 5, 0, 2, 6, 1, 0, 0, 0, 1, 4, 0]])
        for i in range(len(mp_us)):
            uIdx, vIdx = mp_us[i], mp_vs[i]
            J_us.extend([uIdx, vIdx])
            J_vs.extend([vIdx, uIdx])
        J_edges_index = torch.stack([torch.tensor(J_us), torch.tensor(J_vs)], dim=0)

        # . J_edges_type
        J_edges_type = [argmax(e_attr) for i, e_attr in enumerate(J_edges_attr)]
        for i in range(len(mp_us)):
            J_edges_type.extend([4, 4])

        # . J_edges_sym
        J_us_sym = [g2nx.nodes_sym[u] for u in J_us]
        J_vs_sym = [g2nx.nodes_sym[v] for v in J_vs]

        # . J_edges_attr: edge type in one-hot
        expand_edge_attr = label2onehot(labels=torch.tensor(J_edges_type).to(torch.int64), dim=5)
        J_edges_attr = expand_edge_attr 

        # =#######################     construct new pyg-graphs with fake paths     ################################
        pyg = Data()
        pyg.edge_index = J_edges_index
        pyg.edge_attr = J_edges_attr
        pyg.edges_type = torch.tensor(J_edges_type) 
        pyg.x = g2nx.x 
        pyg.y = torch.tensor(g2nx.y, dtype=torch.float32)

        pyg.mp_subgs = uni_subgs  # for check convenience
        pyg.mp_subgs_sym = uni_subgs_sym  # for check convenience

        # =#######################     obtain adj-matrix and adj-lists     ################################
        pyg.edge_attr = J_edges_attr[:,1:]
        pyg_list.append(pyg)

    try:
        torch.save(pyg_list, save_dir+"/{}-ZINC-mp.pkl".format(split))
        print("Metapath construction is done!")
    except (IOError, OSError) as e:
        print("fail to save: ", e)

    print("Metapath process is done!")
    return pyg_list

if __name__ == "__main__":
    # process metapath-construction for dataset
    # pyg_list = load_MP_Dataset(dataset_dir="/data/ZINC"
    #                         , split="train"
    #                         , save_dir="data/Dataset_ZINC/raw")


    current_path = os.path.dirname(__file__)
    dataset_dir = os.path.join(current_path, "../../data/Raw_ZINC")
    save_dir = os.path.join(current_path, "../../data/Dataset_ZINC/raw")

    pyg_list = load_MP_Dataset(dataset_dir=dataset_dir
                            , split="train"
                            , save_dir=save_dir)