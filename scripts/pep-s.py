import torch
from torch.optim import Adam, AdamW
from SubFormer.modules.model import SubFormer
from torch_geometric.datasets import LRGBDataset
from torch_geometric.data import DataLoader
from SubFormer.data.transforms import get_transform
from SubFormer.utils.seed import set_seed
from torch.cuda.amp import GradScaler, autocast

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device='cpu'
set_seed(4321)

root = './dataset_peps'

transform = get_transform(add_virtual_node=False,pedim=16)
train_dataset = LRGBDataset(root, name='Peptides-struct', split='train', pre_transform=transform)
val_dataset = LRGBDataset(root, name='Peptides-struct', split='val', pre_transform=transform)
test_dataset = LRGBDataset(root, name='Peptides-struct', split='test', pre_transform=transform)
train_loader = DataLoader(train_dataset, 64, shuffle=True)
val_loader = DataLoader(val_dataset, 128, shuffle=False)
test_loader = DataLoader(test_dataset, 128, shuffle=False)

epochs = 100
model = SubFormer(
    hidden_channels=64,
    out_channels=11,
    num_mp_layers=2,
    num_enc_layers=3,
    mp_dropout=0.05,
    enc_dropout=0.05,
    local_mp='gine',
    enc_activation='relu',
    aggregation='sum',
    pe_fea=False,
    pe_dim=16,
    n_head=8,
    d_model=128,
    dim_feedforward=128,
    dual_readout=False,
    readout_act='relu',
    expand_spec=False,
    num_eig_trees=32,
    num_eig_graphs=32,
    concat_pe=True,
    signet=False,
    spec_attention=False,
    bypass=True,
).to(device)
print(model)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


print('Number of parameters: ', count_parameters(model))

scaler = GradScaler()


def train():
    model.train()
    total_loss = 0
    for iter, data in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        with autocast():
            out = model(data)
            loss = (out.squeeze() - data.y).abs().mean()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item() * data.num_graphs
    return total_loss / len(train_loader.dataset)

@torch.no_grad()
def test(loader):
    model.eval()
    total_error = 0
    for data in loader:
        data = data.to(device)
        with autocast():
            out = model(data)
            total_error += (out.squeeze() - data.y).abs().mean().item() * data.num_graphs
    return total_error / len(loader.dataset)


optimizer = AdamW(model.parameters(), lr=0.0005, amsgrad=True)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5,
                                                       patience=10, min_lr=0.00001)

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
