import torch
from torch.nn import Linear, Sequential
from torch_geometric.data import Data
from torch_scatter import scatter


class PositionalEncoding(torch.nn.Module):
    def __init__(self,
                 pe_dim: int = 16,
                 hidden_channels: int = 64,
                 activation: str = 'relu',
                 concat_pe: bool = False,
                 signet: bool = False,
                 bypass: bool = False,
                 ):
        super(PositionalEncoding, self).__init__()

        self.concat_pe = concat_pe
        self.signet = signet
        self.bypass = bypass

        if activation is None:
            self.activation = None
        elif activation == 'relu':
            self.activation = torch.nn.ReLU()
        elif activation == 'leaky_relu':
            self.activation = torch.nn.LeakyReLU()
        elif activation == 'gelu':
            self.activation = torch.nn.GELU()

        self.deg_emb = torch.nn.Embedding(100, hidden_channels)
        self.deg_lin = Linear(hidden_channels, hidden_channels)
        self.deg_merge = Linear(hidden_channels, hidden_channels)

        self.tree_lpe_lin = Linear(pe_dim, hidden_channels // 2)
        self.lpe_lin = Linear(pe_dim, hidden_channels // 2)

        if signet:
            self.tree_lpe_lin = Sequential(
                torch.nn.Linear(pe_dim, hidden_channels // 2),
                self.activation,
                torch.nn.Linear(hidden_channels // 2, hidden_channels // 2)
            )
            self.lpe_lin = Sequential(
                torch.nn.Linear(pe_dim, hidden_channels // 2),
                self.activation,
                torch.nn.Linear(hidden_channels // 2, hidden_channels // 2)
            )

    def forward(self, data: Data, x_clique: torch.Tensor) -> torch.Tensor:

        deg = self.deg_emb(data.tree_degree)
        deg = self.deg_lin(deg)
        deg = self.activation(deg)

        x_clique = x_clique + deg
        x_clique = self.deg_merge(x_clique)

        tree_pe = data.tree_lpe.to(torch.float32)
        pe = data.graph_lpe.to(torch.float32)
        pe_mask = torch.isnan(pe)
        pe[pe_mask] = 0
        tree_pe_mask = torch.isnan(tree_pe)
        tree_pe[tree_pe_mask] = 0

        if self.signet:
            tree_pe = self.tree_lpe_lin(tree_pe) + self.tree_lpe_lin(-tree_pe)
            pe = self.lpe_lin(pe) + self.lpe_lin(-pe)

        else:
            if self.bypass:
                tree_pe = self.tree_lpe_lin(tree_pe) + self.tree_lpe_lin(-tree_pe)
                pe = self.lpe_lin(pe) + self.lpe_lin(-pe)
            else:
                tree_pe = self.tree_lpe_lin(tree_pe)
                pe = self.lpe_lin(pe)

        row, col = data.atom2clique_index
        pe = scatter(pe[row], col, dim=0, dim_size=x_clique.size(0), reduce='mean')

        if self.concat_pe:
            x_clique = torch.cat([x_clique, pe, tree_pe], dim=-1)

        else:
            x_clique = x_clique + torch.cat([pe, tree_pe], dim=-1)

        return x_clique
