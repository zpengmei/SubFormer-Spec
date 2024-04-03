import torch
from torch.optim import Adam, AdamW
from SubFormer.modules.model import SubFormer
from tqdm import tqdm
from torch_geometric.data import DataLoader
from SubFormer.data.transforms import get_transform_opda
from SubFormer.utils.seed import set_seed
from SubFormer.datasets.opda import OPDADataset
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
set_seed(4321)

root = './dataset_opda'
target = 0


class MyTransform:
    def __call__(self, data):
        data.y = data.y[target]
        return data


pre_transform = get_transform_opda(add_virtual_node=False, pedim=16)
dataset = OPDADataset(root, pre_transform=pre_transform, transform=MyTransform()).shuffle()

# Normalize targets to mean = 0 and std = 1.
mean = dataset.data.y.mean(dim=0, keepdim=True)
std = dataset.data.y.std(dim=0, keepdim=True)
dataset.data.y = (dataset.data.y - mean) / std
mean, std = mean.item(), std.item()

train_dataset = dataset[:int(len(dataset) * 0.8)]
val_dataset = dataset[int(len(dataset) * 0.8):int(len(dataset) * 0.9)]
test_dataset = dataset[int(len(dataset) * 0.9):]

train_loader = DataLoader(train_dataset, 64, shuffle=True, drop_last=True)
val_loader = DataLoader(val_dataset, 256, shuffle=False, drop_last=True)
test_loader = DataLoader(test_dataset, 256, shuffle=False, drop_last=True)

epochs = 300
model = SubFormer(
    hidden_channels=128,
    out_channels=1,
    num_mp_layers=3,
    num_enc_layers=4,
    enc_dropout=0.05,
    enc_activation='gelu',
    pe_dim=16,
    n_head=8,
    d_model=128,
    dim_feedforward=256,
    dual_readout=False,
    concat_pe=False,
    signet=False,
    spec_attention=False,
    bypass=True,
    num_eig_graphs=16,
    num_eig_trees=16,
    aggregation='sum',
    readout_act='relu',
    pe_fea=False,
    no_spec=False,
).to(device)
print(model)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


print('Number of parameters: ', count_parameters(model))


def train():
    model.train()
    total_loss = 0

    for iter, data in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)
        loss = F.mse_loss(out.squeeze(), data.y)
        loss.backward()
        total_loss += loss.item() * data.num_graphs
        optimizer.step()

    return total_loss / len(train_loader.dataset)


@torch.no_grad()
def test(loader):
    model.eval()
    total_error = 0

    for data in loader:
        data = data.to(device)
        out = model(data)
        total_error += (out.squeeze() * std - data.y * std).abs().sum().item()
    return total_error / len(loader.dataset)


optimizer = Adam(model.parameters(), lr=0.001, amsgrad=True)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5,
                                                       patience=20, min_lr=0.00001)

best_val_mae = test_mae = float('inf')
for epoch in range(1, epochs + 1):
    lr = scheduler.optimizer.param_groups[0]['lr']
    loss = train()
    val_mae = test(val_loader)
    scheduler.step(val_mae)

    if val_mae < best_val_mae:
        best_val_mae = val_mae
        test_mae = test(test_loader)

    print(f'Epoch: {epoch:03d}, LR: {lr:.5f}, Loss: {loss:.4f}, '
          f'Val: {val_mae:.4f}, Test: {test_mae:.4f}')
