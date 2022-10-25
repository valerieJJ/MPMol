import torch
from torch_geometric.datasets import QM9,ZINC,MoleculeNet
from tqdm import tqdm
import os
import os.path as osp
import sys
from typing import Callable, List, Optional
import grandiso
import torch
import torch.nn.functional as F
from torch_scatter import scatter
from torch_geometric.data import (
    Data,
    InMemoryDataset,
    download_url,
    extract_zip,
)
import re
from torch_geometric.utils import from_smiles
import networkx as nx
import pickle
import shutil
from utils import to_network,truncted_BFS,path2mp,mol_paths
HAR2EV = 27.211386246
KCALMOL2EV = 0.04336414
conversion = torch.tensor([
    1., 1., HAR2EV, HAR2EV, HAR2EV, 1., HAR2EV, HAR2EV, HAR2EV, HAR2EV, HAR2EV,
    1., KCALMOL2EV, KCALMOL2EV, KCALMOL2EV, KCALMOL2EV, 1., 1., 1.
])
bonds_dec = {0: '-', 1: '=', 2: '#', 3: '~'}
qm9_node_type={'H': 0, 'C': 1, 'N': 2, 'O': 3, 'F': 4}
qm9_id2ele={v:k for k,v in qm9_node_type.items()}

#zinc_bond_dict = {'NONE':0, 'SINGLE': 1, 'DOUBLE': 2, 'TRIPLE': 3}
zinc_bond_dict = {1: '-', 2: '=', 3: '#'}
zinc_bond_dec = {0:' ', 1: '-', 2: '=', 3: '#'}
zinc_atom_dict = {'C': 0, 'O': 1, 'N': 2, 'F': 3, 'C H1': 4, 'S': 5, 'Cl': 6, 'O -': 7, 'N H1 +': 8, 'Br': 9,
                  'N H3 +': 10, 'N H2 +': 11, 'N +': 12, 'N -': 13, 'S -': 14, 'I': 15, 'P': 16, 'O H1 +': 17,
                  'N H1 -': 18, 'O +': 19, 'S +': 20, 'P H1': 21, 'P H2': 22, 'C H2 -': 23, 'P +': 24, 'S H1 +': 25,
                  'C H1 -': 26, 'P H1 +': 27}
reduce_atom_dict={'C': 'C', 'O': 'O', 'N': 'N', 'F': 'F', 'C H1': 'C', 'S': 'S', 'Cl': 'L', 'O -': 'O', 'N H1 +': 'N', 'Br': 'Br',
                  'N H3 +': 'N', 'N H2 +': 'N', 'N +': 'N', 'N -': 'N', 'S -': 'S', 'I': 'I', 'P': 'P', 'O H1 +': 'O',
                  'N H1 -': 'N', 'O +': 'O', 'S +': 'S', 'P H1': 'P', 'P H2': 'P', 'C H2 -': 'C', 'P +': 'P', 'S H1 +': 'S',
                  'C H1 -': 'C', 'P H1 +': 'P'}
zinc_atom_dict={v:k for k,v in zinc_atom_dict.items()}

with open('metapaths.txt','r') as fin:
    mp=fin.read().split('\n')
    mp= [i.strip('\'').strip(' \'') for i in mp]
    MP_corpus= set(mp)
MP2id={i:k for k,i in enumerate(MP_corpus)}

class MP_Mol(MoleculeNet):
    def process(self):
        with open(self.raw_paths[0], 'r') as f:
            dataset = f.read().split('\n')[1:-1]
            dataset = [x for x in dataset if len(x) > 0]  # Filter empty lines.

        data_list = []
        for line in dataset:
            line = re.sub(r'\".*\"', '', line)  # Replace ".*" strings.
            line = line.split(',')

            smiles = line[self.names[self.name][3]]
            ys = line[self.names[self.name][4]]
            ys = ys if isinstance(ys, list) else [ys]

            ys = [float(y) if len(y) > 0 else float('NaN') for y in ys]
            y = torch.tensor(ys, dtype=torch.float).view(1, -1)

            data = from_smiles(smiles)
            data.y = y
            e_attr=[zinc_bond_dec[int(i)] for i in data.edge_attr]
                #atomx=torch.clip(x,max=7)
            n_attr={k:reduce_atom_dict[zinc_atom_dict[int(i)]] for k,i in enumerate(data.x)}

            nxG=to_network(edge_index,e_attr,n_attr)
            #length=3
            known=set([])
            mp_edges=set([])
            for edges in nxG.edges:
                e1,e2=tuple(edges)

                e1_nei=nx.neighbors(nxG,e1)
                e2_nei=nx.neighbors(nxG,e2)
                for n1 in e1_nei:
                    fmp=(n1,e1,e2)
                    if (fmp not in known) and (fmp[::-1] not in known):
                        known.add(fmp)
                        known.add(fmp[::-1])

                        mp=[nxG.nodes[fmp[0]]['ele'],nxG[fmp[0]][fmp[1]]['attr'],nxG.nodes[fmp[1]]['ele'],\
                            nxG[fmp[1]][fmp[2]]['attr'],nxG.nodes[fmp[2]]['ele']]
                        mp=''.join(mp)
                        if mp in MP_corpus:
                            mp_edges.add((e1,e2,MP2id[mp],3))
                for n2 in e2_nei:
                    fmp=(e1,e2,n2)
                    if (fmp not in known) and (fmp[::-1] not in known):
                        known.add(fmp)
                        known.add(fmp[::-1])
                        mp=[nxG.nodes[fmp[0]]['ele'],nxG[fmp[0]][fmp[1]]['attr'],nxG.nodes[fmp[1]]['ele'],\
                            nxG[fmp[1]][fmp[2]]['attr'],nxG.nodes[fmp[2]]['ele']]
                        mp=''.join(mp)
                        if mp in MP_corpus:
                            mp_edges.add((e1,e2,MP2id[mp],3))
                for n1,n2 in zip(e1_nei,e2_nei):
                    fmp=(n1,e1,e2,n2)
                    if (fmp not in known) and (fmp[::-1] not in known):
                        known.add(fmp)
                        known.add(fmp[::-1])
                        mp=[nxG.nodes[fmp[0]]['ele'],nxG[fmp[0]][fmp[1]]['attr'],nxG.nodes[fmp[1]]['ele'],\
                            nxG[fmp[1]][fmp[2]]['attr'],nxG.nodes[fmp[2]]['ele'],nxG[fmp[2]][fmp[3]]['attr'],\
                            nxG.nodes[fmp[3]]['ele']]
                        mp=''.join(mp)
                        if mp in MP_corpus:
                            mp_edges.add((n1,n2,MP2id[mp],4))
            if len(mp_edges)>0:
                mp_edges=[list(i) for i in mp_edges]
                mp_add=torch.tensor(mp_edges)

                edg_add=mp_add[:,:2].T
                edge_index=torch.cat([edge_index,edg_add],dim=1)
                klp=mp_add[:,3]+4
                edge_attr=torch.cat([edge_attr,klp],dim=0)

            if self.pre_filter is not None and not self.pre_filter(data):
                continue

            if self.pre_transform is not None:
                data = self.pre_transform(data)

            data_list.append(data)

        torch.save(self.collate(data_list), self.processed_paths[0])
class MP_ESOL(MP_Mol):
    def __init__(self, root, transform=None, pre_transform=None,
                 pre_filter=None):
        self.name = 'esol'
        assert self.name in self.names.keys()
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])
class MP_Lipo(MP_Mol):
    def __init__(self, root, transform=None, pre_transform=None,
                 pre_filter=None):
        self.name = 'lipo'
        assert self.name in self.names.keys()
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

class MP_OGBHIV(MP_Mol):
    def __init__(self, root, transform=None, pre_transform=None,
                 pre_filter=None):
        self.name = 'hiv'
        assert self.name in self.names.keys()
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])
class MP_OGBPCBA(MP_Mol):
    def __init__(self, root, transform=None, pre_transform=None,
                 pre_filter=None):
        self.name = 'pcba'
        assert self.name in self.names.keys()
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])
