import torch
from torch.nn import BatchNorm1d, Linear, Sequential, ReLU, Dropout, ModuleList
from torch_geometric.nn.conv import GCNConv, GATConv, GATv2Conv, GINConv, GINEConv
from torch_geometric.nn.aggr import SumAggregation


class baselineGATv2(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout, heads):
        super(baselineGATv2, self).__init__()
        self.dropout = dropout

        self.embed = torch.nn.Embedding(100, hidden_channels)
        self.lins = torch.nn.ModuleList()
        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers - 2):
            self.convs.append(GATv2Conv(hidden_channels, hidden_channels, heads=heads, concat=False))
            self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))

        self.readout = torch.nn.Linear(hidden_channels, out_channels)
        self.readout_sum = SumAggregation()

    def reset_parameters(self):
        self.embed.reset_parameters()
        for lin in self.lins:
            lin.reset_parameters()

        for conv in self.convs:
            conv.reset_parameters()
        self.readout.reset_parameters()

    def forward(self, data):
        x = data.x
        x = self.embed(x)
        edge_index = data.edge_index

        for lin, conv in zip(self.lins, self.convs):
            x = lin(x).relu_()
            x = torch.nn.functional.dropout(x, p=self.dropout, training=self.training)
            x = conv(x, edge_index)

        x = self.readout(x)

        x = self.readout_sum(x, data.batch)

        return x


class baselineGIN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(baselineGIN, self).__init__()
        self.dropout = dropout

        self.embed = torch.nn.Embedding(100, hidden_channels)
        self.norms = torch.nn.ModuleList()
        self.lins = torch.nn.ModuleList()
        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers - 2):
            mlp = Sequential(
                Linear(hidden_channels, hidden_channels),
                BatchNorm1d(hidden_channels),
                ReLU(),
                Linear(hidden_channels, hidden_channels),
            )
            self.convs.append(GINConv(mlp, train_eps=True))
            self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))
            self.norms.append(torch.nn.BatchNorm1d(hidden_channels))

        self.readout = torch.nn.Linear(hidden_channels, out_channels)
        self.readout_sum = SumAggregation()

    def reset_parameters(self):
        self.embed.reset_parameters()
        for lin in self.lins:
            lin.reset_parameters()
        for norm in self.norms:
            norm.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        self.readout.reset_parameters()

    def forward(self, data):
        x = data.x
        x = self.embed(x)
        edge_index = data.edge_index

        for norm, lin, conv in zip(self.norms, self.lins, self.convs):
            x = norm(x)
            x = lin(x).relu_()
            x = torch.nn.functional.dropout(x, p=self.dropout, training=self.training)
            x = conv(x, edge_index)
        x = self.readout(x)

        x = self.readout_sum(x, data.batch)

        return x


class baselineGAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout, heads):
        super(baselineGAT, self).__init__()
        self.dropout = dropout

        self.embed = torch.nn.Embedding(100, hidden_channels * heads)
        self.norms = torch.nn.ModuleList()
        self.lins = torch.nn.ModuleList()
        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden_channels, hidden_channels, heads=heads))
            self.lins.append(torch.nn.Linear(hidden_channels * heads, hidden_channels))
            self.norms.append(torch.nn.BatchNorm1d(hidden_channels))

        self.readout = torch.nn.Linear(hidden_channels, out_channels)
        self.readout_sum = SumAggregation()

    def reset_parameters(self):
        self.embed.reset_parameters()
        for lin in self.lins:
            lin.reset_parameters()
        for norm in self.norms:
            norm.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        self.readout.reset_parameters()

    def forward(self, data):
        x = data.x
        x = self.embed(x)
        edge_index = data.edge_index

        for norm, lin, conv in zip(self.norms, self.lins, self.convs):
            x = norm(x)
            x = lin(x).relu_()
            x = torch.nn.functional.dropout(x, p=self.dropout, training=self.training)
            x, _ = conv(x, edge_index)
        x = self.readout(x)

        x = self.readout_sum(x, data.batch)

        return x


class baselineGCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(baselineGCN, self).__init__()
        self.dropout = dropout

        self.embed = torch.nn.Embedding(100, hidden_channels)
        self.norms = torch.nn.ModuleList()
        self.lins = torch.nn.ModuleList()
        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
            self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))
            self.norms.append(torch.nn.BatchNorm1d(hidden_channels))

        self.readout = torch.nn.Linear(hidden_channels, out_channels)
        self.readout_sum = SumAggregation()

    def reset_parameters(self):
        self.embed.reset_parameters()
        for lin in self.lins:
            lin.reset_parameters()
        for norm in self.norms:
            norm.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        self.readout.reset_parameters()

    def forward(self, data):
        x = data.x
        x = self.embed(x)
        edge_index = data.edge_index

        for norm, lin, conv in zip(self.norms, self.lins, self.convs):
            x = norm(x)
            x = lin(x).relu_()
            x = torch.nn.functional.dropout(x, p=self.dropout, training=self.training)
            x = conv(x, edge_index)
        x = self.readout(x)

        x = self.readout_sum(x, data.batch)

        return x
