import torch
import torch.nn.functional as F
from torch_geometric.utils import get_laplacian, to_scipy_sparse_matrix


def comp_pe(dim: int,
            num_nodes: int,
            edge_attr: torch.Tensor or None,
            edge_index: torch.Tensor,
            normalization='rw',
            use_edge_attr: bool = False,
            is_undirected: bool = True,
            ):
    r"""Computes the local positional encoding of a graph."""
    if normalization not in ['rw', 'sym', None]:
        raise ValueError('Normalization must be either "rw" or "sym" or none')
    if use_edge_attr:
        edge_attr = edge_attr
    else:
        edge_attr = None

    ## get graph laplacian
    edge_index, edge_attr = get_laplacian(edge_index,
                                          edge_attr,
                                          normalization=normalization,
                                          num_nodes=num_nodes)
    ## convert to scipy sparse matrix
    L = to_scipy_sparse_matrix(edge_index, edge_attr, num_nodes)

    # if num_nodes < SPARSE_THRESHOLD:
    from numpy.linalg import eig, eigh
    eig_fn = eig if not is_undirected else eigh

    eig_val, eig_vec = eig_fn(L.todense())  # type: ignore
    eig_val = torch.from_numpy(eig_val).float()
    eig_vec = torch.from_numpy(eig_vec).float()
    eig_vec = torch.real(eig_vec[:, eig_val.argsort()])
    pe = eig_vec[:, 1:dim + 1]

    if pe.shape[1] < dim:
        pe = F.pad(pe, (0, dim - pe.shape[1]), value=float(0))

    assert pe.shape[0] == num_nodes
    assert pe.shape[1] == dim
    assert eig_val.shape[0] == num_nodes

    return pe, eig_val


def comp_deg(
        edge_index: torch.Tensor,
        num_nodes: int,
):
    r"""Computes the degree of each node in a graph."""

    adj = torch.zeros(num_nodes, num_nodes)
    adj[edge_index[0], edge_index[1]] = 1
    degree = adj.long().sum(dim=1).view(-1)

    return degree
