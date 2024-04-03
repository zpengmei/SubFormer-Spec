import torch
from torch.optim import Adam, AdamW
from tqdm import tqdm
from torch.optim.lr_scheduler import CosineAnnealingLR
from SubFormer.modules.model import SubFormer
from torch_geometric.datasets import LRGBDataset
from torch_geometric.data import DataLoader
from SubFormer.data.transforms import get_transform
from SubFormer.utils.seed import set_seed
from torch.cuda.amp import autocast, GradScaler
from sklearn.metrics import average_precision_score
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
set_seed(4321)

root = 'dataset_pepf'
transform = get_transform(add_virtual_node=False, pedim=16)
train_dataset = LRGBDataset(root, name='Peptides-func', split='train', pre_transform=transform)
val_dataset = LRGBDataset(root, name='Peptides-func', split='val', pre_transform=transform)
test_dataset = LRGBDataset(root, name='Peptides-func', split='test', pre_transform=transform)
train_loader = DataLoader(train_dataset, 64, shuffle=True)
val_loader = DataLoader(val_dataset, 64, shuffle=False)
test_loader = DataLoader(test_dataset, 64, shuffle=False)

total_epochs = 200
warmup_epochs = 20
lr = 0.0005
model = SubFormer(
    hidden_channels=96,
    out_channels=10,
    num_mp_layers=3,
    num_enc_layers=3,
    mp_dropout=0.05,
    enc_dropout=0.2,
    local_mp='gine',
    enc_activation='gelu',
    readout_act='gelu',
    aggregation='sum',
    num_eig_trees=32,
    num_eig_graphs=64,
    pe_fea=False,
    pe_dim=16,
    n_head=8,
    d_model=96,
    dim_feedforward=96,
    dual_readout=False,
    spec_attention=True,
    expand_spec=True,
    concat_pe=False,
    signet=False,
    bypass=True,
).to(device)
print(model)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


print('Number of parameters: ', count_parameters(model))


def eval_ap(y_true, y_pred):
    ap_list = []
    y_true = y_true.detach().cpu().numpy()
    y_pred = y_pred.detach().cpu().numpy()

    for i in range(y_true.shape[1]):
        if np.sum(y_true[:, i] == 1) > 0 and np.sum(y_true[:, i] == 0) > 0:
            is_labeled = y_true[:, i] == y_true[:, i]
            ap = average_precision_score(y_true[is_labeled, i], y_pred[is_labeled, i])
            ap_list.append(ap)

    if len(ap_list) == 0:
        raise RuntimeError('No positively labeled data available. Cannot compute Average Precision.')
    return sum(ap_list) / len(ap_list)


scaler = GradScaler()


def train():
    model.train()
    y_preds, y_trues = [], []
    total_loss = 0
    for data in tqdm(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        mask = ~torch.isnan(data.y)
        with autocast():
            out = model(data)[mask].reshape(-1, 10)
            y = data.y.to(torch.float)[mask].reshape(-1, 10)
            y_preds.append(out)
            y_trues.append(y)
            loss = torch.nn.BCEWithLogitsLoss()(out, y)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item() * data.num_graphs
    y_preds = torch.cat(y_preds, dim=0)
    y_trues = torch.cat(y_trues, dim=0)
    train_perf = eval_ap(y_true=y_trues.cpu(), y_pred=y_preds.cpu())
    return total_loss / len(train_loader.dataset), train_perf


@torch.no_grad()
def test(loader):
    model.eval()

    y_preds, y_trues = [], []
    total_loss = 0
    for data in loader:  # Use loader instead of train_loader
        data = data.to(device)
        mask = ~torch.isnan(data.y)
        with autocast():
            out = model(data)[mask].reshape(-1, 10)
            y = data.y.to(torch.float)[mask].reshape(-1, 10)

            y_trues.append(y)
            y_preds.append(out)

            loss = torch.nn.BCEWithLogitsLoss()(out, y)
            total_loss += loss.item() * data.num_graphs

    y_preds = torch.cat(y_preds, dim=0)
    y_trues = torch.cat(y_trues, dim=0)
    test_perf = eval_ap(y_true=y_trues.cpu(), y_pred=y_preds.cpu())
    return total_loss / len(loader.dataset), test_perf


def adjust_learning_rate(optimizer, epoch, warmup_epochs=50, base_lr=0.001):
    lr = base_lr * (epoch / warmup_epochs)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


optimizer = AdamW(model.parameters(), lr=lr, amsgrad=True, weight_decay=1e-8)
scheduler_cosine = CosineAnnealingLR(optimizer, T_max=total_epochs - warmup_epochs)

best_val_mae = test_mae = float('inf')
best_val_ap = test_ap = 0

for epoch in range(1, total_epochs + 1):
    if epoch <= warmup_epochs:
        adjust_learning_rate(optimizer, epoch, warmup_epochs, base_lr=lr)
    else:
        scheduler_cosine.step()

    lr = optimizer.param_groups[0]['lr']
    loss, train_ap = train()

    val_mae, val_ap = test(val_loader)  # Receive two outputs from test function

    if val_mae < best_val_mae:
        best_val_mae = val_mae
        best_val_ap = val_ap  # Update best_val_ap
    test_mae, test_ap = test(test_loader)

    print(f'Epoch: {epoch:03d}, LR: {lr:.5f}, Loss: {loss:.4f}, '
          f'Train AP: {train_ap:.4f}, '  # Print Train AP
          f'Val MAE: {val_mae:.4f}, Val AP: {val_ap:.4f}, '  # Print Val AP
          f'Test MAE: {test_mae:.4f}, Test AP: {test_ap:.4f}')  # Print Test AP
