
import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.data import DataLoader
from torch_geometric.utils import remove_self_loops
from models import Network
from qm9_preprocess.DatasetQM9 import DatasetQM9
from tqdm import tqdm

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

transform = T.Compose([
    Complete(),
    T.Distance(norm=False)])

target = 0
node_dim = 8
hidden=64
edge_dim=6
nheads=8
dropout=0.2
recurs=3
lr=0.001
factor=0.7
patience=5
min_lr=0.00001
epochs=1000
batch_size=64
num_workers=16
ckpt_dir = "save"
data_path = "../data/Dataset_QM9"
model_path = '{}/MPMol-QM9-target-{}.pkl'.format(ckpt_dir, target)
device = torch.device('cuda:0')

dataset = DatasetQM9(data_path, transform=transform).shuffle()
mean, std = dataset.data.y.mean(dim=0, keepdim=True), dataset.data.y.std(dim=0, keepdim=True)
dataset.data.y = (dataset.data.y - mean) / std
dataset.data.y = dataset.data.y[:, target]
mean, std = mean[:, target].item(), std[:, target].item()
test_dataset, val_dataset, train_dataset = dataset[:10000], dataset[10000:20000], dataset[20000:]
test_loader = DataLoader(test_dataset, batch_size, shuffle=False) # DataLoader(test_dataset, batch_size, num_workers, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size, shuffle=False)
train_loader = DataLoader(train_dataset, batch_size, shuffle=True)

model = Network(infeat=node_dim, dim=hidden, edge_dim=edge_dim, nheads=nheads, dropout=dropout, recurs=recurs)
model.to(device)
loss_fc = F.mse_loss
optimizer = torch.optim.Adam(model.parameters(), lr)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor, patience
                                                       , min_lr)


print("target:{}, dataset:{}, loss_fc:{} ".format(target, data_path, loss_fc))
print("model:\n", model)

def train(epoch):
    model.train()
    mse_loss_all = 0
    with tqdm(train_loader) as tq:
        for iter, data in enumerate(tq):
            tq.set_description('Iter %d' % iter)
            data = data.to(device)
            optimizer.zero_grad()
            logits = model(data)
            loss = loss_fc(logits, data.y)
            loss.backward()
            mse_loss_all += loss.item() * data.num_graphs
            optimizer.step()
    return mse_loss_all / len(train_loader.dataset)

def test(loader):
    model.eval()
    mae_loss_all = 0
    mse_loss_all = 0
    for data in loader:
        data = data.to(device)
        logits = model(data)  
        mae_loss_all += (logits * std - data.y * std).abs().sum().item() 
        mse_loss = F.mse_loss(logits*std, data.y*std)
        mse_loss_all += mse_loss.item() * data.num_graphs
    return mse_loss_all / len(loader.dataset), mae_loss_all / len(loader.dataset)

best_val_error = None
best_test_error = None
for epoch in range(0, epochs):
    lr = scheduler.optimizer.param_groups[0]['lr']
    train_mse_loss = train(epoch)
    val_mse_error, val_mae_error = test(val_loader)
    scheduler.step(val_mae_error)
    test_mse_error, test_mae_error = test(test_loader)

    if best_test_error is None or best_test_error > test_mae_error:
        best_test_error = test_mae_error
        # torch.save(model.state_dict(), model_path)

    print('Epoch:{}, Train-MSE:{:.5f}, Valid MSE:{:.5f}, Valid MAE:{:.5f} '
          'Test MAE: {:.5f}, best-test-mae:{:.5f} target-{}, devide:{}, model-save:{}'
          .format(epoch, train_mse_loss, val_mse_error, val_mae_error, test_mae_error, best_test_error, target, device, model_path))
