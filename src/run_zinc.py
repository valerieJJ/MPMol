import torch
import torch.nn.functional as F
from torch_geometric.data import DataLoader
from zinc_preprocess.DatasetZINC import DatasetZINC
from src.models import Network
from tqdm import tqdm

node_dim = 28
hidden = 64
edge_dim=4
nheads=8
recurs=3
lr=0.001
factor=0.7
patience=5
min_lr=0.00001
epochs=1000
batch_size=64
num_workers=16
ckpt_dir = "save"
rootpath = "data/Dataset_ZINC" 
device = torch.device('cuda:0')

train_dataset = DatasetZINC(root=rootpath, split="train")
test_dataset = DatasetZINC(root=rootpath, split="test")
val_dataset = DatasetZINC(root=rootpath, split="val")

y1 = train_dataset.data.y
y2 = val_dataset.data.y
y3 = test_dataset.data.y
y = torch.cat([y1, y2, y3],dim=0)
mean, std = y.mean(dim=0, keepdim=True), y.std(dim=0, keepdim=True)

test_dataset.data.y = (test_dataset.data.y - mean) / std
train_dataset.data.y = (train_dataset.data.y - mean) / std
val_dataset.data.y = (val_dataset.data.y - mean) / std
test_loader = DataLoader(test_dataset, batch_size, num_workers, shuffle=False)  # , pin_memory=True
val_loader = DataLoader(val_dataset, batch_size, num_workers, shuffle=False)
train_loader = DataLoader(train_dataset, batch_size, num_workers, shuffle=True)

model = Network(infeat=node_dim, dim=hidden, edge_dim=edge_dim, nheads=nheads, recurs=recurs)
model.to(device)
loss_fc = F.mse_loss
print("dataset:{}, loss_fc:{} ".format(rootpath, loss_fc))
print("model:\n", model)
optimizer = torch.optim.Adam(model.parameters(), lr)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                       factor, # 0.7
                                                       patience, # 5
                                                       min_lr, mode='min')

def train(epoch):
    model.train()
    loss_all = 0
    mae_error = 0
    with tqdm(train_loader) as tq:
        for iter, data in enumerate(tq):
            tq.set_description('Iter %d' % iter)
            data = data.to(device)
            optimizer.zero_grad()
            logits = model(data)
            loss = loss_fc(logits, data.y)
            mae_error += (logits * std.cuda() - data.y * std.cuda()).abs().sum().item()  # MAE
            loss.backward()
            loss_all += loss.item() * data.num_graphs
            optimizer.step()
    return loss_all / len(train_loader.dataset), mae_error/len(train_loader.dataset)

def test(loader):
    model.eval()
    error = 0
    mse_loss_all = 0
    for data in loader:
        data = data.to(device)
        logits = model(data)
        error += (logits * std.cuda() - data.y * std.cuda()).abs().sum().item()
        mse_loss = F.mse_loss(logits, data.y)
        mse_loss_all += mse_loss.item() * data.num_graphs
    return mse_loss_all / len(loader.dataset), error / len(loader.dataset)

best_val_error = None
best_test_error = None
for epoch in range(1, epochs):
    lr = scheduler.optimizer.param_groups[0]['lr']
    train_mse_loss, train_mae_loss = train(epoch)
    val_mse_error, val_mae_error = test(val_loader)
    scheduler.step(val_mae_error)
    test_mse_error, test_mae_error = test(test_loader)

    if best_val_error is None or val_mae_error < best_val_error:
        best_val_error = val_mae_error

    if best_test_error is None or best_test_error > test_mae_error:
        best_test_error = test_mae_error
        torch.save(model.state_dict(), '{}/MPMol-ZINC.pkl'.format(ckpt_dir))

    print('Epoch: {:03d}, LR: {:7f}, Train-MSE-Loss: {:.7f}, Validation MAE: {:.7f}, '
          'Test MAE: {:.7f}, best-test:{} save-dir:{}, device:{}'
          .format(epoch, lr, train_mse_loss, val_mae_error, test_mae_error, best_test_error, device))
