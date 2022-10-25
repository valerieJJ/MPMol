from queue import Queue
import networkx as nx
import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.data import DataLoader
from torch_geometric.utils import remove_self_loops
import pysmiles
def to_network(edge_index,edge_attr,node_attr,direct=True):
    if direct:
        G=nx.Graph()
    else:
        G=nx.DiGraph()
    if edge_index.size(0)==2:

        edge_index=edge_index.T.tolist()
    assert len(edge_index)==len(edge_attr)
    for k,i in enumerate(edge_index):
        e1,e2=tuple(i)
        G.add_edge(e1,e2)

        G[e1][e2]['attr']=edge_attr[k]
    
    for n in G.nodes:
        G.nodes[n]['ele']=node_attr[n]
    return G
def truncted_BFS(G,source,depth=3):
    paths=[[source]]

    que=Queue()
    travesed_node=set([source])

    que.put(source)
    #current_depth=[0]
    while not que.empty():
        n=que.get()
        t=list(nx.neighbors(G,n))
        if len(t)>0 and len(paths[-1])<=depth:
            #qwer=current_depth[-1]
            for j in t:
                if j not in travesed_node:
                    que.put(j)
                    #current_depth.append(qwer+1)
                    add_paths=[]
                    for y in paths:
                        if y[-1]==n:
                            dd=y.copy()
                            dd.append(j)
                            add_paths.append(dd)
                    paths.extend(add_paths)
                    travesed_node.add(j)
    while len(paths[-1])>depth:
        paths.pop()
    return paths
def path2mp(G,paths):
    metapaths={3:[],4:[],5:[]}
    if isinstance(paths,list):
        for i in paths:
            l=len(i)
            if l>0 and l in metapaths:
                mp=[G.nodes[i[0]]['ele']]
                last_ele=i[0]
                for p in i[1:]:
                    mp.append(G[last_ele][p]['attr'])
                    mp.append(G.nodes[p]['ele'])
                    last_ele=p
                mp=''.join(mp)
                metapaths[l].append(mp)
    elif isinstance(paths,dict):
        for k,v in paths.items():

            for i in v:
                mp=[G.nodes[i[0]]['ele']]
                last_ele=i[0]
                for p in i[1:]:
                    mp.append(G[last_ele][p]['attr'])
                    mp.append(G.nodes[p]['ele'])
                    last_ele=p
                mp=''.join(mp)
                metapaths[k].append(mp)
    return metapaths
def mol_paths(G,depth=4):
    res={3:set([]),4:set([]),5:set([])}
    for i in G.nodes:
        paths=truncted_BFS(G,i,depth=depth)
        for p in paths:

            if len(p) in res:

                res[len(p)].add(tuple(p))
    
    return res

def pyg2smiles(data):
    pass



class Complete(object):
    def __call__(self, data):
        device = data.edge_index.device

        row = torch.arange(data.num_nodes, dtype=torch.long, device=device)
        col = torch.arange(data.num_nodes, dtype=torch.long, device=device)

        row = row.view(-1, 1).repeat(1, data.num_nodes).view(-1)
        col = col.repeat(data.num_nodes)
        edge_index = torch.stack([row, col], dim=0)

        edge_attr = None
        if data.edge_attr is not None:
            idx = data.edge_index[0] * data.num_nodes + data.edge_index[1]
            size = list(data.edge_attr.size())
            size[0] = data.num_nodes * data.num_nodes
            edge_attr = data.edge_attr.new_zeros(size)
            edge_attr[idx] = data.edge_attr

        edge_index, edge_attr = remove_self_loops(edge_index, edge_attr)
        data.edge_attr = edge_attr
        data.edge_index = edge_index

        return data

class MyTransform(object):
    def __call__(self, data):
        data.y = data.y[:, target]
        return data


if __name__=='__main__':

    pass
    import networkx as nx


    G=nx.Graph()

    G.add_edge(1,'tg')

    G.add_edge(1,'rerr')
    G.add_edge('rerr','tg')

    nx.draw(G)