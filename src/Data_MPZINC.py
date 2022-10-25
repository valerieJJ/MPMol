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


class MP_ZINC(ZINC):
    def process(self):
        for split in ['train', 'val', 'test']:
            with open(osp.join(self.raw_dir, f'{split}.pickle'), 'rb') as f:
                mols = pickle.load(f)

            indices = range(len(mols))

            if self.subset:
                with open(osp.join(self.raw_dir, f'{split}.index'), 'r') as f:
                    indices = [int(x) for x in f.read()[:-1].split(',')]

            pbar = tqdm(total=len(indices))
            pbar.set_description(f'Processing {split} dataset')

            data_list = []
            for idx in indices:
                mol = mols[idx]

                x = mol['atom_type'].to(torch.long).view(-1, 1)
                y = mol['logP_SA_cycle_normalized'].to(torch.float)

                adj = mol['bond_type']
                edge_index = adj.nonzero(as_tuple=False).t().contiguous()
                edge_attr = adj[edge_index[0], edge_index[1]].to(torch.long)

                e_attr=[zinc_bond_dec[int(i)] for i in edge_attr]
                #atomx=torch.clip(x,max=7)
                n_attr={k:reduce_atom_dict[zinc_atom_dict[int(i)]] for k,i in enumerate(x)}

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
                
                
                data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr,
                            y=y)

                if self.pre_filter is not None and not self.pre_filter(data):
                    continue

                if self.pre_transform is not None:
                    data = self.pre_transform(data)

                data_list.append(data)
                pbar.update(1)

            pbar.close()

            torch.save(self.collate(data_list),
                       osp.join(self.processed_dir, f'{split}.pt'))