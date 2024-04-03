import torch
from torch.nn import TransformerEncoder, TransformerEncoderLayer, ReLU
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.utils import to_dense_batch


class Encoder(torch.nn.Module):

    def __init__(self,
                 d_model: int,
                 n_head: int,
                 num_encoder_layers: int,
                 dim_feedforward: int,
                 dropout: float,
                 activation: str,
                 spec_attention: bool = False,
                 num_eig_graphs: int = 1,
                 num_eig_trees: int = 1,
                 nospec: bool = False,
                 expand_spec: bool = False,
                 ):
        super().__init__()

        self.d_model = d_model
        self.nhead = n_head
        self.num_encoder_layers = num_encoder_layers
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.activation = ReLU()
        self.nospec = nospec
        self.expand_spec = expand_spec
        self.spec_attention = spec_attention

        self.encoder_layer = TransformerEncoderLayer(d_model, n_head, dim_feedforward, dropout, activation,batch_first=True)
        self.encoder = TransformerEncoder(self.encoder_layer, num_layers=num_encoder_layers)

        if not nospec:
            total_eig = num_eig_graphs + num_eig_trees
            self.total_eig = total_eig
            self.graph_eig = num_eig_graphs
            self.tree_eig = num_eig_trees

            if expand_spec:
                self.cls_token = torch.nn.Linear(total_eig, d_model)
                self.value_token = torch.nn.Linear(d_model, d_model)
                self.time_constant = torch.nn.Linear(d_model, d_model)

            else:
                self.cls_token = torch.nn.Parameter(torch.randn(total_eig, d_model))
                self.value_token = torch.nn.Parameter(torch.randn(total_eig, d_model))
                self.time_constant = torch.nn.Parameter(torch.randn(total_eig))

        else:
            self.cls_token = torch.nn.Parameter(torch.randn(1, d_model))

    def forward(self, x_clique: torch.Tensor, data: Data) -> torch.Tensor:
        tree_batch = torch.repeat_interleave(data.num_cliques)
        src, mask = to_dense_batch(x_clique, batch=tree_batch)

        if not self.nospec:
            graphval = data.graph_lpeval.squeeze()
            treeval = data.tree_lpeval.squeeze()
            graphval, _ = to_dense_batch(graphval, batch=data.batch)
            graphval = F.pad(graphval, (0, self.graph_eig - graphval.shape[1]), value=0)
            treeval, _ = to_dense_batch(treeval, batch=tree_batch)
            treeval = F.pad(treeval, (0, self.tree_eig - treeval.shape[1]), value=0)
            val = torch.cat([graphval, treeval], dim=-1)

            if self.expand_spec:
                val = self.cls_token(val).tanh()
                xsq = torch.pow(self.time_constant(val), 2)
                eig_graphs = (1 - xsq) * torch.exp(-xsq / 2)
                eig_vals = eig_graphs
                eig_token = F.softmax(eig_vals, dim=-1)
                vals = self.value_token(eig_vals).relu()

            else:
                xsq = torch.pow(val * self.time_constant, 2)
                eig_graphs = (1 - xsq) * torch.exp(-xsq / 2)
                eig_vals = eig_graphs
                eig_token = F.softmax(torch.matmul(eig_vals, self.cls_token), dim=-1)
                vals = torch.matmul(eig_vals, self.value_token).relu()

            if self.spec_attention:
                cls_token = vals * eig_token
            else:
                cls_token = vals

            cls_token = cls_token.unsqueeze(1)

        else:
            cls_token = self.cls_token.expand(src.shape[0], -1, -1)

        src = torch.cat([cls_token, src], dim=1)
        mask = torch.cat([torch.ones(src.shape[0], 1).bool().to(src.device), mask], dim=1)
        output = self.encoder(src, src_key_padding_mask=~mask)
        output = output[:, 0, :]

        return output
