from locale import strcoll
import os.path as osp

import torch
import torch.nn.functional as F
from torch.nn import GRU, Linear, ReLU, Sequential

import torch_geometric.transforms as T
from torch_geometric.datasets import ModelNet
from torch_geometric.loader import DataLoader
from torch_geometric.nn import NNConv, Set2Set,GCNConv,GIN,GatedGraphConv
from torch_geometric.utils import remove_self_loops
from Data_MP_MOL import MP_ESOL,MP_OGBHIV,MP_OGBPCBA,MP_Lipo
from Data_MPZINC import MP_ZINC
from Data_MPQM9 import MP_QM9
from models import MPMol
import sys
import argparse
parser=argparse.ArgumentParser()
parser.add_argument('--dataset',default='qm9',type=str)
parser.add_argument('--t',default=0,type=int)
parser.add_argument('--dim',type=int,default=64)
parser.add_argument('--epoch',default=2,type=int)
parser.add_argument('--o',type=str,default='runs')
parser.add_argument('--model',default='mpmol')
parser.add_argument('--gpu',default='0',type=str)
parser.add_argument('--recurs',defualt=3,type=int)
parser.add_argument('--nheads',type=int,default=8)
parser.add_argument('--dropout',type=float,default=0.5)
args=parser.parse_args()
target = args.t
dim = args.dim

class MyTransform(object):
    def __call__(self, data):
        data.y = data.y[:, target]
        return data

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

if args.dataset.lower()=='qm9':
    path ='dataset/QM9'
    transform = T.Compose([MyTransform(), Complete(), T.Distance(norm=False)])
    dataset = MP_QM9(path, transform=transform).shuffle()
elif args.dataset.lower()=='zinc':
    path ='dataset/zinc_origin'
    transform = T.Compose([MyTransform(), Complete(), T.Distance(norm=False)])
    dataset = MP_ZINC(path, transform=transform).shuffle()
elif args.dataset.lower()=='mutag':
    path='dataset/MUTAG'
    transform = T.Compose([MyTransform(), Complete(), T.Distance(norm=False)])
    dataset = ModelNet(path,'MUTAG', transform=transform).shuffle()
elif args.dataset.lower()=='nci':
    path='dataset/NCI1'
    transform = T.Compose([MyTransform(), Complete(), T.Distance(norm=False)])
    dataset = ModelNet(path,'nci', transform=transform).shuffle()
elif args.dataset.lower()=='ptc':
    path='dataset/PTC_FR'
    transform = T.Compose([MyTransform(), Complete(), T.Distance(norm=False)])
    dataset = ModelNet(path,'ptc', transform=transform).shuffle()
elif args.dataset.lower()=='hiv':
    path='dataset/hiv'
    transform = T.Compose([MyTransform(), Complete(), T.Distance(norm=False)])
    dataset = MP_OGBHIV(path,'ptc', transform=transform).shuffle()
elif args.dataset.lower()=='pcba':
    path='dataset/pcba'
    transform = T.Compose([MyTransform(), Complete(), T.Distance(norm=False)])
    dataset = MP_OGBPCBA(path,'ptc', transform=transform).shuffle()



mean = dataset.data.y.mean(dim=0, keepdim=True)
std = dataset.data.y.std(dim=0, keepdim=True)
dataset.data.y = (dataset.data.y - mean) / std
mean, std = mean[:, target].item(), std[:, target].item()
l_data=len(dataset.data)

test_dataset = dataset[:int(l_data*0.1)]
val_dataset = dataset[int(l_data*0.1):int(l_data*0.2)]
train_dataset = dataset[int(l_data*0.2):]
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)


if torch.cuda.is_available() and args.gpu!='cpu':
    device='cuda:'+args.gpu
else:
    device='cpu'
device=torch.device(device)
if args.model.lower()=='mpmol':
    model=model = MPMol(infeat=dataset.node_dim, dim=args.hidden, edge_dim=dataset.edge_dim, nheads=args.nheads, dropout=args.dropout, recurs=args.recurs)


optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                       factor=0.7, patience=5,
                                                       min_lr=0.00001)


def train():
    model.train()
    loss_all = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        loss = F.mse_loss(model(data), data.y)
        loss.backward()
        loss_all += loss.item() * data.num_graphs
        optimizer.step()
    return loss_all / len(train_loader.dataset)


def test(loader):
    model.eval()
    absolute_error = 0
    std_error=0
    for data in loader:
        data = data.to(device)
        stand_y_pred=model(data)
        absolute_error += (stand_y_pred * std - data.y * std).abs().sum().item()  # MAE

        std_error+=(stand_y_pred-data.y).abs().sum().item()
    return absolute_error / len(loader.dataset),std_error/ len(loader.dataset)

best_val_error = None
save_name='|%s|target%d|ep%d|dim%d|'%(args.model,args.t,args.epoch,args.dim)
outfile=open(osp.join(args.o,save_name+'.txt'),'w')
print('mean:%.4f,std:%.4f'%(mean,std),file=outfile,end='\n')
for epoch in range(1, args.epoch):
    lr = scheduler.optimizer.param_groups[0]['lr']
    loss = train(epoch)
    val_error,_ = test(val_loader)
    scheduler.step(val_error)
    if best_val_error is None or val_error <= best_val_error:
        test_mae,test_stdmae = test(test_loader)
        best_val_error = val_error
        prt=f'Epoch: {epoch:03d}, LR: {lr:7f}, Loss: {loss:.7f}, '+f'Val MAE: {val_error:.7f}, Test MAE: {test_mae:.7f}, Test stdMAE: {test_stdmae:.7f}'
        print(prt,file=outfile,end='\n')
        print(prt)
outfile.close()
save_path=osp.join('save_model','checkpoint_'+save_name)
torch.save(model,save_path+'.pt')