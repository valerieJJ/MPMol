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
with open('metapaths.txt','r') as fin:
    mp=fin.read().split('\n')
    mp= [i.strip('\'').strip(' \'') for i in mp]
    MP_corpus= set(mp)
MP2id={i:k for k,i in enumerate(MP_corpus)}
class MP_QM9(QM9):
    def process(self):
        try:
            import rdkit
            from rdkit import Chem, RDLogger
            from rdkit.Chem.rdchem import BondType as BT
            from rdkit.Chem.rdchem import HybridizationType
            RDLogger.DisableLog('rdApp.*')

        except ImportError:
            rdkit = None
        if rdkit is None:
            print(("Using a pre-processed version of the dataset. Please "
                   "install 'rdkit' to alternatively process the raw data."),
                  file=sys.stderr)

            data_list = torch.load(self.raw_paths[0])
            data_list = [Data(**data_dict) for data_dict in data_list]

            if self.pre_filter is not None:
                data_list = [d for d in data_list if self.pre_filter(d)]

            if self.pre_transform is not None:
                data_list = [self.pre_transform(d) for d in data_list]

            torch.save(self.collate(data_list), self.processed_paths[0])
            return

        types = {'H': 0, 'C': 1, 'N': 2, 'O': 3, 'F': 4}
        bonds = {BT.SINGLE: 0, BT.DOUBLE: 1, BT.TRIPLE: 2, BT.AROMATIC: 3}

        with open(self.raw_paths[1], 'r') as f:
            target = f.read().split('\n')[1:-1]
            target = [[float(x) for x in line.split(',')[1:20]]
                      for line in target]
            target = torch.tensor(target, dtype=torch.float)
            target = torch.cat([target[:, 3:], target[:, :3]], dim=-1)
            target = target * conversion.view(1, -1)

        with open(self.raw_paths[2], 'r') as f:
            skip = [int(x.split()[0]) - 1 for x in f.read().split('\n')[9:-2]]

        suppl = Chem.SDMolSupplier(self.raw_paths[0], removeHs=False,
                                   sanitize=False)
        data_list = []
        for i, mol in enumerate(tqdm(suppl)):
            if i in skip:
                continue

            N = mol.GetNumAtoms()

            conf = mol.GetConformer()
            pos = conf.GetPositions()
            pos = torch.tensor(pos, dtype=torch.float)

            type_idx = []
            atomic_number = []
            aromatic = []
            sp = []
            sp2 = []
            sp3 = []
            num_hs = []
            for atom in mol.GetAtoms():
                type_idx.append(types[atom.GetSymbol()])
                atomic_number.append(atom.GetAtomicNum())
                aromatic.append(1 if atom.GetIsAromatic() else 0)
                hybridization = atom.GetHybridization()
                sp.append(1 if hybridization == HybridizationType.SP else 0)
                sp2.append(1 if hybridization == HybridizationType.SP2 else 0)
                sp3.append(1 if hybridization == HybridizationType.SP3 else 0)

            z = torch.tensor(atomic_number, dtype=torch.long)

            row, col, edge_type = [], [], []
            for bond in mol.GetBonds():
                start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                row += [start, end]
                col += [end, start]
                edge_type += 2 * [bonds[bond.GetBondType()]]

            edge_index = torch.tensor([row, col], dtype=torch.long)
            edge_type = torch.tensor(edge_type, dtype=torch.long)
            edge_attr = F.one_hot(edge_type,
                                  num_classes=len(bonds)).to(torch.float)

            perm = (edge_index[0] * N + edge_index[1]).argsort()
            edge_index = edge_index[:, perm]
            edge_type = edge_type[perm]
            edge_attr = edge_attr[perm]

            row, col = edge_index
            hs = (z == 1).to(torch.float)
            num_hs = scatter(hs[row], col, dim_size=N).tolist()

            x1 = F.one_hot(torch.tensor(type_idx), num_classes=len(types))
            x2 = torch.tensor([atomic_number, aromatic, sp, sp2, sp3, num_hs],
                              dtype=torch.float).t().contiguous()
            x = torch.cat([x1.to(torch.float), x2], dim=-1)

            y = target[i].unsqueeze(0)
            name = mol.GetProp('_Name')
            e_attr=[bonds_dec[int(ii)] for ii in edge_attr.argmax(dim=1)]
            n_attr={kk:qm9_id2ele[int(ii)] for kk,ii in enumerate(x[:,:5].argmax(dim=1))}

            nxG=to_network(edge_index,e_attr,n_attr)
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
                klp=mp_add[:,2]+edge_attr.size(1)
                
                edge_attr=edge_attr.argmax(dim=1)
                edge_attr=torch.cat([edge_attr,klp],dim=0)

            else:
                edge_attr=edge_attr.argmax(dim=1)


            data = Data(x=x, z=z, pos=pos, edge_index=edge_index,
                        edge_attr=edge_attr, y=y, name=name, idx=i)

            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)
            data_list.append(data)
        torch.save(self.collate(data_list), self.processed_paths[0])