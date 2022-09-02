import os
import torch
from numpy.core.fromnumeric import argmax
from torch_geometric.utils import to_networkx
from torch_geometric.data import Data
from tqdm import tqdm
from torch_geometric.datasets.qm9 import QM9

bonds_dec = {0:'-',1:'=',2:'#',3:'~'}

def walks_from_one_node(start_nodeIdx, nodes_sym, adj_list, walk_length):
        queue = []
        visited = []
        kpaths = [] # depth=k
        allpaths = []
        path_str_syb = []
        kpaths_syb = []
        allpaths_syb = []

        queue.append(start_nodeIdx)
        visited.append(start_nodeIdx)
        path_str = []
        deep = 0
        k = walk_length

        while len(queue)!=0 :            
            if deep<=k:
                centerIdx = queue.pop(0)   
                if len(path_str)!=0:
                    idx = int(path_str[-1])
                    nbs_id = adj_list[idx].keys()           
                    if centerIdx in nbs_id:
                        if len(queue)!=0:
                            bond = queue.pop(0)
                            path_str.append(bond)
                            path_str_syb.append(bond)
                        path_str.append(str(centerIdx))
                        path_str_syb.append(nodes_sym[centerIdx]) 
                        allpaths.append("".join(path_str))
                        allpaths_syb.append("".join(path_str_syb))
                    else:
                        if len(queue)!=0:
                            bond = queue.pop(0)
                            continue
                else:
                    if len(queue)!=0:
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

                centerIdx_flag = 0 #- 当前中心点的可走邻居数: =0则没有新节点进栈, =n则有n个节点入栈
                for neibIdx in adj_list[centerIdx].keys():
                    if (neibIdx not in visited) and (deep<k):
                        bondtype_onehot = adj_list[centerIdx][neibIdx]['edge_attr']
                        bondtype_sym = bonds_dec[argmax(bondtype_onehot)] #- 取出与该邻居节点之间的BondType# 0-single,1-double,2-triple,3-aromatic
                        queue.append(neibIdx) #- 邻居节点id先入栈
                        queue.append(bondtype_sym)#- 然后再将BondType入栈
                        centerIdx_flag = centerIdx_flag + 1 #- 记录在center-node的邻居中，新入栈的邻居节点的个数

                if  deep == k or centerIdx_flag == 0: #- 没有新节点进栈，或路径长度==k
                    if len(path_str)==0:
                        print("path_str==0  ||||||  ",start_nodeIdx)
                    if len(path_str)>1:
                        path_str.pop() # nodeidx
                        path_str.pop() # bond
                        path_str_syb.pop() #- 先将nb-id出栈
                        path_str_syb.pop() #- 再将与nb-id相连的BondType出栈
                        deep = deep - 1 #- 路径长度 减1
                        
                    else :
                        if len(path_str)==1:
                            path_str.pop() # nodeidx
                            path_str_syb.pop() #- 先将nb-id出栈
                            deep = deep - 1 #- 路径长度 减1

        return allpaths,allpaths_syb, kpaths, kpaths_syb

def getMetapathPairs(start_nodesIdx, nodes_sym,  metapath_copus, adj_list, walk_length):
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
        for i,path in enumerate(paths):
            reversed_path = path[::-1]
            if path not in totalpaths:#+ and (int(path[0]) != int(path[-1])): #~ 去除重复和自环
                if (int(path[0]) != int(path[-1])) or (int(path[0])==int(path[-1]) and (len(path)>1)):
                    totalpaths.append(path)
                    totalpaths_syb.append(paths_syb[i])
                    if paths_syb[i] in metapath_copus:
                        find_mp.append(path)
                        find_mp_symbol.append(paths_syb[i])
                        mp_us.append(int(path[0]))
                        mp_vs.append(int(path[-1]))

    return mp_us, mp_vs, find_mp, find_mp_symbol

def label2onehot(labels, dim): #~ Convert label indices to one-hot vectors.
        out = torch.zeros(list(labels.size()) + [dim])
        out.scatter_(len(out.size())-1, labels.unsqueeze(-1), 1.)
        return out

def get_Iterable_adj_dict(adj_list):   
    g_adj_list = {}
    for a in adj_list:
        center_nodeIdx, neighbors_dict = a
        g_adj_list[center_nodeIdx] = neighbors_dict #- construct an iterable adj-list
    return g_adj_list

def uniqueMPSubgraphs(mp_subgs,mp_subgs_sym):
    lst = mp_subgs
    mp_subgs_sym
    for i,ele in enumerate(lst):
        rev = ele[::-1]
        cnt = lst.count(rev)
        while (rev in lst) and (lst.count(rev)>0):
            idx = lst.index(rev)
            lst.remove(rev)
            mp_subgs_sym.pop(idx)
    return lst,mp_subgs_sym

def load_MP_Dataset(dataset_dir, save_dir):
    qm9 = QM9(root=dataset_dir)
    # qm9.process()
    # nodes_symbols = qm9.symbols #~ {'H': 1, 'C': 6, 'N': 7, 'O': 8, 'F': 9}
    nodes_symbols = {'H': 1, 'C': 6, 'N': 7, 'O': 8, 'F': 9}
    bonds_types = qm9.bonds #~ {rdkit.Chem.rdchem.BondType.SINGLE: 0, rdkit.Chem.rdchem.BondType.DOUBLE: 1, rdkit.Chem.rdchem.BondType.TRIPLE: 2, rdkit.Chem.rdchem.BondType.AROMATIC: 3}
    bonds_dec = {bonds_types[key]:key for key in bonds_types.keys()}
    bonds_dec[4] = 'mp-edge'
    
    nodes_symbols_dec = {nodes_symbols[key]:key for key in nodes_symbols.keys()}
    pyg_list = []
    metapath_copus = [
        "C=C-C-O","O-C-C=C",
        "C#C-C-O","O-C-C#C",
        "C-C=O","O=C-C",
        "O-C=O","O=C-O",
        "C-O-C=O","O=C-O-C",
        "C#C-C","C-C#C",
        "N#C-C","C-C#N",
        "O-C-N","N-C-O",
        "N-C=O","O=C-N",
        "C-O-C",
        "C~C~C",
        "O-N-C", "C-N-O",
        "C-O-O-C"]

    for i,g in enumerate(tqdm(qm9, desc="mp-processing")):
        g = qm9[i] 
        x = torch.cat((g.x[:,1:5], g.x[:,6:-1]),dim=1) # ignore atom H
        y = g.y.numpy().tolist().copy() #. y is the graph target label
        y = [y[0]]
        x = x.numpy().tolist().copy()
        z = g.z.numpy().tolist().copy()
        g_nodes_sym = []
        for i,atomic_num in enumerate(z):
            g_nodes_sym.append(nodes_symbols_dec[atomic_num])

        #=#######################     obtain adj-matrix and adj-list     ################################
        g2nx = to_networkx(data=g,edge_attrs=['edge_attr'],to_undirected=True)
        adj_list = g2nx.adjacency() #- the returned adj-list is disiterable
        g_adj_list = get_Iterable_adj_dict(adj_list=adj_list)
        
        #=#######################     obtain adj-matrix and adj-list     ################################
        start_nodesIdx = []
        for i,symbol in enumerate(g_nodes_sym):
          start_nodesIdx.append(i)
        walk_length = 4
        if len(z) < walk_length:
          walk_length = len(z) 
        mp_us, mp_vs, find_mp, find_mp_symbol = getMetapathPairs(start_nodesIdx=start_nodesIdx, 
                                                                    nodes_sym=g_nodes_sym, 
                                                                    metapath_copus=metapath_copus,
                                                                    adj_list=g_adj_list, 
                                                                    walk_length=walk_length)
        uni_subgs, uni_subgs_sym = uniqueMPSubgraphs(find_mp,find_mp_symbol)
        mp_us = []
        mp_vs = []
        for ele in uni_subgs:
            mp_us.append(int(ele[0]))
            mp_vs.append(int(ele[-1]))
        #=#######################   copy graph-info & adding the fake paths based on metapaths     ################################
        edge_attr = g.edge_attr# one-hot edge type
        edge_index = g.edge_index
        J_x = x 
        J_y = y
        J_z = z

        J_edges_attr = edge_attr.numpy().tolist().copy()
        J_edges_type = {i:argmax(e_attr)+1 for i,e_attr in enumerate(J_edges_attr)}
        J_us = edge_index[0].numpy().tolist().copy()
        J_vs = edge_index[1].numpy().tolist().copy()
        J_nodes_sym = g_nodes_sym
        
        for i in range(len(mp_us)):
            uIdx, vIdx = mp_us[i], mp_vs[i]
            J_us.extend([uIdx, vIdx])
            J_vs.extend([vIdx, uIdx])
        J_edges_index = torch.stack([torch.tensor(J_us),torch.tensor(J_vs)],dim=0)

        #. J_edges_type 
        J_edges_type = [argmax(e_attr) for i,e_attr in enumerate(J_edges_attr)]
        for i in range(len(mp_us)): 
            J_edges_type.extend([4,4])
        
        #. J_edges_sym
        J_us_sym = [J_z[u] for u in J_us]
        J_vs_sym = [J_z[v] for v in J_vs]
        J_edges_sym = [J_us_sym, J_vs_sym]

        #. J_edges_attr: edge type in one-hot
        expand_edge_attr = label2onehot(labels=torch.tensor(J_edges_type).to(torch.int64)
                                        , dim=5)
        J_edges_attr = expand_edge_attr 
        
        #=#######################     construct new pyg-graphs with fake paths     ################################
        pyg = Data()
        pyg.edge_index = J_edges_index
        pyg.edge_attr = J_edges_attr 
        pyg.edges_type = torch.tensor(J_edges_type) # eg. len=14,[1, 1, 1, 1, 1, 2, 1, 2, 1, 1, 1, 1, 5, 5]
        pyg.edges_sym = J_edges_sym 
        pyg.nodes_sym = J_nodes_sym 
        
        pyg.x = torch.tensor(J_x)
        pyg.y = torch.tensor(J_y,dtype=torch.float32) 
        pyg.z = J_z
        #=#######################     obtain adj-matrix and adj-lists     ################################
        pyg.pos = g.pos
        pyg_list.append(pyg)
    
    try:
        torch.save(pyg_list, save_dir)
        print("Metapath construction is done!")
    except (IOError, OSError) as e:
        print("fail to save: ", e)

    return pyg_list


if __name__=="__main__":
    # process metapath-construction for dataset
    current_path = os.path.dirname(__file__)
    dataset_dir = os.path.join(current_path, "../../data/Raw_QM9")
    save_dir = os.path.join(current_path, "../../data/Dataset_QM9/raw/mp_qm9.pkl")

    pyg_list = load_MP_Dataset(dataset_dir=dataset_dir
                            , save_dir=save_dir)