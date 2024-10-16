import torch
from torch_geometric.data import Data
from SubFormer.modules.local_mp import LocalMP
from SubFormer.modules.pe import PositionalEncoding
from SubFormer.modules.encoder import Encoder


class SubFormer(torch.nn.Module):
    '''
    Main SubFormer model.
    '''

    def __init__(self,
                 ### Local MP part ###
                 hidden_channels: int = 64,
                 num_mp_layers: int = 4,
                 mp_dropout: float = 0,
                 local_mp: str = 'gine',
                 aggregation: str = 'sum',
                 no_tree: bool = False,
                 return_graph: bool = False,
                 ### Transformer part ###
                 num_enc_layers: int = 4,
                 enc_dropout: float = 0,
                 enc_activation: str = 'relu',
                 n_head: int = 8,
                 d_model: int = 128,
                 dim_feedforward: int = 512,
                 ### Positional Encoding part ###
                 concat_pe: bool = False,
                 signet: bool = False,
                 pe_fea: bool = False,
                 pe_dim: int = 10,
                 pe_activation: str = 'gelu',
                 bypass: bool = False,
                 pe_source: str = 'both',
                 ### Spectrum part ###
                 spec_attention: bool = False,
                 num_eig_graphs: int = 16,
                 num_eig_trees: int = 16,
                 no_spec: bool = False,
                 expand_spec: bool = False,
                 gate_activation: str = 'relu',
                 ### readout part ###
                 readout_act: str = None,
                 readout_num_layers: int = 2,
                 readout_channels: int = 64,
                 dual_readout: bool = False,
                 out_channels: int = 1,
                 ):
        super(SubFormer, self).__init__()

        self.local_mp = LocalMP(hidden_channels=hidden_channels,
                                out_channels=hidden_channels,
                                num_layers=num_mp_layers,
                                dropout=mp_dropout,
                                local_mp=local_mp,
                                aggregation=aggregation,
                                pe_fea=pe_fea,
                                pe_dim=pe_dim,
                                no_tree=no_tree,
                                return_graph=return_graph,
                                )

        self.pe = PositionalEncoding(pe_dim=pe_dim,
                                     hidden_channels=hidden_channels,
                                     concat_pe=concat_pe,
                                     signet=signet,
                                     activation=pe_activation,
                                     bypass=bypass,
                                     pe_source=pe_source,
                                     )

        self.encoder = Encoder(d_model=d_model,
                               n_head=n_head,
                               num_encoder_layers=num_enc_layers,
                               dim_feedforward=dim_feedforward,
                               dropout=enc_dropout,
                               activation=enc_activation,
                               spec_attention=spec_attention,
                               num_eig_graphs=num_eig_graphs,
                               num_eig_trees=num_eig_trees,
                               nospec=no_spec,
                               expand_spec=expand_spec,
                               gate_activation=gate_activation,
                               )

        if readout_act is None:
            raise ValueError('Please provide a readout activation function.')
        elif readout_act == 'relu':
            self.activation = torch.nn.ReLU()
        elif readout_act == 'leaky_relu':
            self.activation = torch.nn.LeakyReLU()
        elif readout_act == 'gelu':
            self.activation = torch.nn.GELU()

        self.dual_readout = dual_readout

        if dual_readout:

            # add num_layers for readout
            self.readout = torch.nn.Sequential()
            for i in range(readout_num_layers):
                if i == 0:
                    self.readout.add_module(f'linear_{i}', torch.nn.Linear(d_model + hidden_channels, readout_channels))
                    self.readout.add_module(f'activation_{i}', self.activation)
                elif i == readout_num_layers - 1:
                    self.readout.add_module(f'linear_{i}', torch.nn.Linear(readout_channels, out_channels))
                else:
                    self.readout.add_module(f'linear_{i}', torch.nn.Linear(readout_channels, readout_channels))
                    self.readout.add_module(f'activation_{i}', self.activation)


        else:

            self.readout = torch.nn.Sequential()
            for i in range(readout_num_layers):
                if i == 0:
                    self.readout.add_module(f'linear_{i}', torch.nn.Linear(d_model, readout_channels))
                    self.readout.add_module(f'activation_{i}', self.activation)
                elif i == readout_num_layers - 1:
                    self.readout.add_module(f'linear_{i}', torch.nn.Linear(readout_channels, out_channels))
                else:
                    self.readout.add_module(f'linear_{i}', torch.nn.Linear(readout_channels, readout_channels))
                    self.readout.add_module(f'activation_{i}', self.activation)


    def forward_notree(self, data: Data):
        x, graph_readout = self.local_mp(data)
        x = self.pe.forward_notree(x=x, data=data)
        out = self.encoder.forward_notree(x=x, data=data)

        if self.dual_readout:
            out = torch.concat((out, graph_readout), dim=1)
        else:
            out = out

        out = self.readout(out)
        return out

    def forward(self, data: Data):
        x_clique, graph_readout = self.local_mp(data)
        x_clique = self.pe(x_clique=x_clique, data=data)
        tree_out = self.encoder(x_clique=x_clique, data=data)

        if self.dual_readout:
            out = torch.concat((tree_out, graph_readout), dim=1)
        else:
            out = tree_out

        out = self.readout(out)
        return out
