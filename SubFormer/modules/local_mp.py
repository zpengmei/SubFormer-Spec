import torch
from SubFormer.modules.agat import AGATConv
import torch.nn.functional as F
from torch.nn import Sequential, Linear, BatchNorm1d
from torch_geometric.nn import GINEConv, ResGatedGraphConv
from torch_geometric.data import Data
from torch_scatter import scatter


class AtomEncoder(torch.nn.Module):
    def __init__(self, hidden_channels):
        super(AtomEncoder, self).__init__()

        self.embeddings = torch.nn.ModuleList()

        for i in range(9):
            self.embeddings.append(torch.nn.Embedding(100, hidden_channels))

    def reset_parameters(self):
        for embedding in self.embeddings:
            embedding.reset_parameters()

    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(1)

        out = 0
        for i in range(x.size(1)):
            out += self.embeddings[i](x[:, i])
        return out


class BondEncoder(torch.nn.Module):
    def __init__(self, hidden_channels):
        super(BondEncoder, self).__init__()

        self.embeddings = torch.nn.ModuleList()

        for i in range(3):
            self.embeddings.append(torch.nn.Embedding(6, hidden_channels))

    def reset_parameters(self):
        for embedding in self.embeddings:
            embedding.reset_parameters()

    def forward(self, edge_attr):
        if edge_attr.dim() == 1:
            edge_attr = edge_attr.unsqueeze(1)

        out = 0
        for i in range(edge_attr.size(1)):
            out += self.embeddings[i](edge_attr[:, i])
        return out


class LocalMP(torch.nn.Module):
    def __init__(self,
                 hidden_channels: int = 64,
                 out_channels: int = 64,
                 num_layers: int = 3,
                 dropout: float = 0,
                 local_mp: str = 'gine',
                 aggregation: str = 'sum',
                 pe_fea: bool = False,
                 pe_dim: int = 10,
                 ):
        super(LocalMP, self).__init__()
        self.atom_encoder = AtomEncoder(hidden_channels)
        self.clique_encoder = torch.nn.Embedding(4, hidden_channels)
        self.num_layers = num_layers
        self.aggregation = aggregation

        self.bond_encoders = torch.nn.ModuleList()
        self.graph_convs = torch.nn.ModuleList()
        self.graph_norms = torch.nn.ModuleList()
        self.sub_norms = torch.nn.ModuleList()

        if local_mp == 'gine':
            if pe_fea:
                for i in range(num_layers):
                    if i == 0:
                        self.bond_encoders.append(BondEncoder(hidden_channels))
                        nn = Sequential(
                            Linear(hidden_channels + pe_dim, 2 * hidden_channels),
                            BatchNorm1d(2 * hidden_channels),
                            torch.nn.ReLU(),
                            Linear(2 * hidden_channels, hidden_channels),
                        )
                        conv = GINEConv(nn=nn,
                                        train_eps=True,
                                        edge_dim=hidden_channels,
                                        )
                        self.graph_convs.append(conv)
                        self.graph_norms.append(BatchNorm1d(hidden_channels))
                        self.sub_norms.append(BatchNorm1d(hidden_channels))
                    else:
                        self.bond_encoders.append(BondEncoder(hidden_channels))
                        nn = Sequential(
                            Linear(hidden_channels, 2 * hidden_channels),
                            BatchNorm1d(2 * hidden_channels),
                            torch.nn.ReLU(),
                            Linear(2 * hidden_channels, hidden_channels),
                        )
                        conv = GINEConv(nn=nn,
                                        train_eps=True,
                                        )
                        self.graph_convs.append(conv)
                        self.graph_norms.append(BatchNorm1d(hidden_channels))
                        self.sub_norms.append(BatchNorm1d(hidden_channels))
            else:
                for i in range(num_layers):
                    self.bond_encoders.append(BondEncoder(hidden_channels))
                    nn = Sequential(
                        Linear(hidden_channels, 2 * hidden_channels),
                        BatchNorm1d(2 * hidden_channels),
                        torch.nn.ReLU(),
                        Linear(2 * hidden_channels, hidden_channels),
                    )
                    conv = GINEConv(nn=nn,
                                    train_eps=True,
                                    )
                    self.graph_convs.append(conv)
                    self.graph_norms.append(BatchNorm1d(hidden_channels))
                    self.sub_norms.append(BatchNorm1d(hidden_channels))

        elif local_mp == 'agat':
            if pe_fea:
                self.align = torch.nn.Linear(hidden_channels + pe_dim, hidden_channels)
                for i in range(num_layers):
                    if i == 0:
                        self.bond_encoders.append(BondEncoder(hidden_channels))
                        conv = AGATConv(in_channels=hidden_channels + pe_dim, edge_dim=hidden_channels)
                        self.graph_convs.append(conv)
                        self.graph_norms.append(BatchNorm1d(hidden_channels))
                        self.sub_norms.append(BatchNorm1d(hidden_channels))
                    else:
                        self.bond_encoders.append(BondEncoder(hidden_channels))
                        conv = AGATConv(hidden_channels, edge_dim=hidden_channels)
                        self.graph_convs.append(conv)
                        self.graph_norms.append(BatchNorm1d(hidden_channels))
                        self.sub_norms.append(BatchNorm1d(hidden_channels))
            else:
                for _ in range(num_layers):
                    self.bond_encoders.append(BondEncoder(hidden_channels))
                    conv = AGATConv(hidden_channels, edge_dim=hidden_channels)
                    self.graph_convs.append(conv)
                    self.graph_norms.append(BatchNorm1d(hidden_channels))
                    self.sub_norms.append(BatchNorm1d(hidden_channels))


        elif local_mp == 'ggcn':
            if pe_fea:
                self.align = torch.nn.Linear(hidden_channels + pe_dim, hidden_channels)
                for i in range(num_layers):
                    if i == 0:
                        self.bond_encoders.append(BondEncoder(hidden_channels))
                        conv = ResGatedGraphConv(hidden_channels, hidden_channels, edge_dim=hidden_channels + pe_dim)
                        self.graph_convs.append(conv)
                        self.graph_norms.append(BatchNorm1d(hidden_channels))
                        self.sub_norms.append(BatchNorm1d(hidden_channels))
                    else:
                        self.bond_encoders.append(BondEncoder(hidden_channels))
                        conv = ResGatedGraphConv(hidden_channels, hidden_channels, edge_dim=hidden_channels)
                        self.graph_convs.append(conv)
                        self.graph_norms.append(BatchNorm1d(hidden_channels))
                        self.sub_norms.append(BatchNorm1d(hidden_channels))
            else:
                for _ in range(num_layers):
                    self.bond_encoders.append(BondEncoder(hidden_channels))
                    conv = ResGatedGraphConv(hidden_channels, hidden_channels, edge_dim=hidden_channels)
                    self.graph_convs.append(conv)
                    self.graph_norms.append(BatchNorm1d(hidden_channels))
                    self.sub_norms.append(BatchNorm1d(hidden_channels))

        else:
            raise NotImplementedError

        self.local_mp = local_mp

        self.atom2clique_lins = torch.nn.ModuleList()
        self.clique2atom_lins = torch.nn.ModuleList()

        for _ in range(num_layers):
            self.atom2clique_lins.append(
                Linear(hidden_channels, hidden_channels))
            self.clique2atom_lins.append(
                Linear(hidden_channels, hidden_channels))

        self.clique = Linear(hidden_channels, out_channels)

        self.dropout = dropout
        self.pe_fea = pe_fea

    def forward(self, data: Data):

        x = self.atom_encoder(data.x.squeeze())
        x_clique = self.clique_encoder(data.x_clique.squeeze())
        x_clique = self.clique(x_clique)

        graph_pe = data.graph_lpe.float()
        pe_mask = torch.isnan(graph_pe)
        graph_pe[pe_mask] = 0

        for i in range(self.num_layers):
            edge_attr = self.bond_encoders[i](data.edge_attr_graph)
            edge_index = data.edge_index_graph

            if self.pe_fea:
                if i == 0:
                    x_positive = torch.cat([x, graph_pe], dim=-1)
                    x_negative = torch.cat([x, -graph_pe], dim=-1)
                    x = self.graph_convs[i](x_positive, edge_index, edge_attr) + self.graph_convs[i](x_negative,
                                                                                                     edge_index,
                                                                                                        edge_attr)
                    if self.local_mp != 'gine':
                        x = self.align(x)
                    x = self.graph_norms[i](x).relu()
                    x = F.dropout(x, p=self.dropout)

                else:
                    x = self.graph_convs[i](x, edge_index, edge_attr)
                    x = self.graph_norms[i](x).relu()
                    x = F.dropout(x, p=self.dropout)

            else:
                x = self.graph_convs[i](x, edge_index, edge_attr)
                x = self.graph_norms[i](x).relu()
                x = F.dropout(x, p=self.dropout)


            row, col = data.atom2clique_index
            x_clique = x_clique + F.relu(self.atom2clique_lins[i](scatter(
                x[row], col, dim=0, dim_size=x_clique.size(0),
                reduce=self.aggregation)))
            x_clique = self.sub_norms[i](x_clique)
            x = x + F.leaky_relu(self.clique2atom_lins[i](scatter(
                x_clique[col], row, dim=0, dim_size=x.size(0),
                reduce=self.aggregation)))

        graph_readout = scatter(x, data.batch, reduce='sum', dim=0)

        return x_clique, graph_readout
