import torch
from torch.optim import Adam, AdamW
from SubFormer.modules.model import SubFormer as SF
from torch_geometric.datasets import ZINC
from torch_geometric.data import DataLoader
from SubFormer.data.transforms import get_transform_zinc
from SubFormer.utils.seed import set_seed
from torch.optim.lr_scheduler import CosineAnnealingLR

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
set_seed(4321)

root = './dataset_zinc'
transform = get_transform_zinc(pedim=16)
val_dataset = ZINC(root, subset=True, split='val', pre_transform=transform)
train_dataset = ZINC(root, subset=True, split='train', pre_transform=transform)
test_dataset = ZINC(root, subset=True, split='test', pre_transform=transform)
train_loader = DataLoader(train_dataset, 32, shuffle=True)
val_loader = DataLoader(val_dataset, 1000, shuffle=False)
test_loader = DataLoader(test_dataset, 1000, shuffle=False)

# Total number of epochs and warm-up epochs
total_epochs = 1000
warmup_epochs = 50

model = SF(
    hidden_channels=64,
    out_channels=1,
    num_mp_layers=2,
    num_enc_layers=3,
    mp_dropout=0,
    enc_dropout=0.1,
    local_mp='gine',
    enc_activation='relu',
    readout_act='relu',
    aggregation='sum',
    pe_fea=False,
    pe_dim=16,
    n_head=8,
    d_model=128,
    dim_feedforward=128,
    spec_attention=True,
    expand_spec=False,
    num_eig_graphs=16,
    num_eig_trees=16,
    concat_pe=True,
    signet=True,
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
        loss = (out.squeeze() - data.y).abs().mean()

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
        total_error += (out.squeeze() - data.y).abs().sum().item()
    return total_error / len(loader.dataset)


def adjust_learning_rate(optimizer, epoch, warmup_epochs=50, base_lr=0.001):
    lr = base_lr * (epoch / warmup_epochs)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


optimizer = Adam(model.parameters(), lr=0.001)
scheduler_cosine = CosineAnnealingLR(optimizer, T_max=total_epochs - warmup_epochs)

best_val_mae = test_mae = float('inf')
for epoch in range(1, total_epochs + 1):
    if epoch <= warmup_epochs:
        adjust_learning_rate(optimizer, epoch, base_lr=0.001)
    else:
        scheduler_cosine.step()

    lr = optimizer.param_groups[0]['lr']
    loss = train(epoch)
    val_mae = test(val_loader)

    if val_mae < best_val_mae:
        best_val_mae = val_mae
        test_mae = test(test_loader)

    print(f'Epoch: {epoch:03d}, LR: {lr:.5f}, Loss: {loss:.4f}, '
          f'Val: {val_mae:.4f}, Test: {test_mae:.4f}')
