from scipy.linalg.interpolative import seed
import torch
import torch.nn as nn
from torch_geometric.nn.conv import GATConv,GCNConv
from torch.nn import Sequential, Linear, ReLU, GRU,Dropout
from torch_geometric.nn import NNConv, Set2Set
import torch.nn.functional as F
from numpy.core.fromnumeric import argmax
from torch_sparse.tensor import to


def getMaskedEdges(edge_index, edge_attr, eType):
    c = torch.sum(edge_attr, dim=1)
    edge_maskkk = c!=0
    masked_edges = edge_attr[edge_maskkk]
    masked_idx = edge_index[:, edge_maskkk]
    edge_types = torch.argmax(masked_edges, dim=1)
    eType_mask2 = edge_types==eType
    get_edges_idx = masked_idx[:, eType_mask2]
    return get_edges_idx

class RelationalAttention(nn.Module):
    def __init__(self, in_feat, edges_num, activation_fc):
        super(RelationalAttention, self).__init__()

        self.edges_num = edges_num
        self.importance_fn = nn.Sequential(
            nn.Linear(in_features=in_feat, out_features=2 * in_feat, bias=True),
            nn.ReLU(),
            nn.Linear(in_features=2 * in_feat, out_features=1, bias=False)
        )
        self.activatin_fc = activation_fc

    def forward(self, emb):
        rela_importance = self.importance_fn(emb) 
        rela_importance = torch.mean(rela_importance, dim=0) 
        rela_weights = torch.softmax(rela_importance, dim=0) 
        nodes_embs = emb * rela_weights
        nodes_embs_out = nodes_embs.sum(1)
        return nodes_embs_out

class MolAtt(nn.Module):
    def __init__(self, in_feat, out_feat, etypes, n_heads, dropout=0.2, activation_fc=None, residual=True):
        super(MolAtt, self).__init__()

        self.etypes = etypes
        self.residual = residual

        self.multi_gat_layers = nn.ModuleList()
        for etype in range(self.etypes):
            self.multi_gat_layers.append(GATConv(in_feat, out_feat, heads=n_heads, concat=True))
        self.relaAttnlayer = RelationalAttention(in_feat=out_feat * n_heads, edges_num=self.etypes,
                                                 activation_fc=activation_fc)
        self.dropout = nn.Dropout(dropout)
        self.activation_fc = activation_fc
        self.batchnorm_h = nn.BatchNorm1d(out_feat * n_heads)

    def forward(self, h, batch):
        emb = h
        output = []
        for etype in range(self.etypes):
            masked_edgeIndex = getMaskedEdges(batch.edge_index, batch.edge_attr[:, :self.etypes], eType=etype)
            etype_h = self.multi_gat_layers[etype](x=emb, edge_index=masked_edgeIndex.to(torch.long))
            output.append(etype_h)
        output = torch.stack(output, 1)
        output = self.relaAttnlayer(output)
        output = F.elu(output)
        return output

class MPMol(torch.nn.Module):
    def __init__(self, infeat=8, dim=64, edge_dim=6, nheads=8, dropout=0.0, recurs=3):
        super(MPMol, self).__init__()
        self.lin0 = torch.nn.Linear(infeat, dim) 
        self.layers = torch.nn.ModuleList()
        self.edge_dim = edge_dim # 6 
        self.recurs = recurs
        nn = Sequential(
            Linear(self.edge_dim, 128),
            ReLU(),
            Linear(128, dim * dim)
        )
        self.dropout = Dropout(dropout)
        self.layers.append(MolAtt(in_feat=dim, out_feat=int(dim / nheads), etypes=self.edge_dim,
                                          n_heads=nheads, dropout=dropout, activation_fc=F.relu))
        self.layers.append(NNConv(dim, dim, nn, aggr='mean'))
        self.gru = GRU(dim, dim)
        self.set2set = Set2Set(dim, processing_steps=5) 
        self.lin1 = torch.nn.Linear(2 * dim, dim)
        self.lin2 = torch.nn.Linear(dim, 1)

    def forward(self, data):
        out = F.relu(self.lin0(data.x))
        h = out.unsqueeze(0)
        for recur in range(self.recurs):
            out = F.relu(self.layers[0](out, data)) 
            out = self.dropout(out)
            m = F.relu(self.layers[1](out, data.edge_index, data.edge_attr)) 
            out, h = self.gru(m.unsqueeze(0), h)
            out = out.squeeze(0)
        out = self.set2set(out, data.batch)
        out = F.relu(self.lin1(out))
        out = self.lin2(out)
        return out.view(-1)

class GCN(torch.nn.Module):
    def __init__(self,intdim,dim):
        super().__init__()
        self.lin0 = torch.nn.Linear(intdim, dim)

        self.conv1=GCNConv(dim,dim)
        self.dp1=torch.nn.Dropout(0.5)
        self.conv2=GCNConv(dim,dim)


        self.set2set = Set2Set(dim, processing_steps=3)
        self.lin1 = torch.nn.Linear(2 * dim, dim)
        self.lin2 = torch.nn.Linear(dim, 1)
    def forward(self, data):
        out = F.relu(self.lin0(data.x))
        x=self.conv1(out, data.edge_index)
        x=self.dp1(x)
        x=torch.relu(x)
        x=self.conv2(x, data.edge_index)
        out = self.set2set(out, data.batch)
        out = F.relu(self.lin1(out))
        out = self.lin2(out)
        return out.view(-1)
class GIN(torch.nn.Module):
    def __init__(self,intdim,dim):
        super().__init__()
        self.lin0 = torch.nn.Linear(intdim, dim)

        self.conv1=GIN(dim,dim)
        self.dp1=torch.nn.Dropout(0.5)
        self.conv2=GIN(dim,dim)


        self.set2set = Set2Set(dim, processing_steps=3)
        self.lin1 = torch.nn.Linear(2 * dim, dim)
        self.lin2 = torch.nn.Linear(dim, 1)
    def forward(self, data):
        out = F.relu(self.lin0(data.x))
        x=self.conv1(out, data.edge_index)
        x=self.dp1(x)
        x=torch.relu(x)
        x=self.conv2(x, data.edge_index)
        out = self.set2set(out, data.batch)
        out = F.relu(self.lin1(out))
        out = self.lin2(out)
        return out.view(-1)
