# Similar to Junction tree class, but this one is used in the GTR code, can be useful when not dealing with molecules.

import torch
import networkx as nx
from scipy.sparse.csgraph import minimum_spanning_tree
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx, to_scipy_sparse_matrix, from_scipy_sparse_matrix, to_undirected

from itertools import chain
from utils import comp_deg


class SkeletonTreeData(Data):
    def __inc__(self, key, item, *args):
        if key == 'tree_edge_index':
            return self.x_clique.size(0)
        elif key == 'atom2clique_index':
            return torch.tensor([[self.x.size(0)], [self.x_clique.size(0)]])
        else:
            return super(SkeletonTreeData, self).__inc__(key, item, *args)


class SkeletonTree(object):
    def __call__(self, data):
        tree_edge_index, node2bag_index, num_bags, x_bag, _ = tree_decomposition_bag(data)

        data = SkeletonTreeData(**{k: v for k, v in data})

        data.tree_edge_index = tree_edge_index  # edge index in the skeleton tree
        data.atom2clique_index = node2bag_index  # map each node in the original graph to the bag(s) in the skeleton tree
        data.num_cliques = num_bags  # number of bags in the skeleton tree
        data.x_clique = x_bag  # identifier for each bag (0: clique, 1: cycle, 2: edge, 3: isolated node or singleton)
        data.tree_degree = comp_deg(data.tree_edge_index, data.num_cliques)
        data.edge_attr_graph = data.edge_attr
        data.edge_index_graph = data.edge_index
        if data.x is None:
            data.x = position_features(data.edge_index, data.num_nodes, 16)
        return data


def position_features(edge_index, num_nodes, pos_dim):
    if edge_index.size(1) == 0:
        features = torch.zeros(num_nodes, pos_dim)
    else:  # random walk process
        A = torch.sparse.FloatTensor(edge_index, torch.ones(edge_index.size(1)), torch.Size([num_nodes, num_nodes]))

        idx = torch.LongTensor([range(num_nodes), range(num_nodes)])
        elem = torch.sparse.sum(A, dim=-1).to_dense().clamp(min=1).pow(-1)
        D_inv = torch.sparse.FloatTensor(idx, elem, torch.Size([num_nodes, num_nodes])).to_dense()  # D^-1

        # iteration
        M_power = torch.sparse.mm(A, D_inv)
        M = M_power.to_sparse()
        features = list()
        for i in range(2 * pos_dim - 1):
            M_power = torch.sparse.mm(M, M_power)
            if i % 2 == 0:
                features.append(torch.diagonal(M_power))

        features = torch.stack(features, dim=-1)

    return features


def tree_decomposition_bag(data):
    # print(data)
    graph_data = Data(edge_index=data.edge_index, num_nodes=data.num_nodes)
    G = to_networkx(graph_data, to_undirected=True, remove_self_loops=True)

    # process isolated nodes and generate their corresponding bags
    bags = [[i] for i in list(nx.isolates(G))]
    x_bag = [3] * len(bags)
    # remove isolated nodes
    G.remove_nodes_from(list(nx.isolates(G)))

    # process cliques and generate their corresponding bags
    cliques = list()
    for c in nx.find_cliques(G):
        if len(c) >= 3:  # filter out edges, and they will be considered later in the 'edges'
            cliques.append(c)
            bags.append(c)
            x_bag.append(0)

    # process cycles and generate their corresponding bags
    cycles = list()
    for c in nx.cycle_basis(G):
        if len(c) >= 4:  # filter out triangles, since they have already been considered in the 'cliques'
            cycles.append(c)
            bags.append(c)
            x_bag.append(1)

    # remove cliques
    for c in cliques:
        S = G.subgraph(c)
        clique_edges = list(S.edges)
        G.remove_edges_from(clique_edges)
        G.remove_nodes_from(list(nx.isolates(G)))
    # remove cycles
    for c in cycles:
        cycle_edges = list(zip(c[:-1], c[1:]))
        cycle_edges.append((c[-1], c[0]))
        G.remove_edges_from(cycle_edges)
        G.remove_nodes_from(list(nx.isolates(G)))

    # process edges and generate their corresponding bags
    edges = list()
    for e in G.edges:
        edges.append(list(e))
        bags.append(list(e))
        x_bag.append(2)

    # generate the 'node2bag' maps
    node2bag = [[] for i in range(data.num_nodes)]
    for b in range(len(bags)):
        for node in bags[b]:
            node2bag[node].append(b)

    # add singleton bags (the intersection of at least 3 bags) and construct the edges of the skeleton graph
    w_edge = {}
    for node in range(data.num_nodes):
        b_node = node2bag[node]
        if len(b_node) <= 1:  # 'node' cannot be a singleton bag according to its definition
            continue

        edges, strcs, weights = list(), list(), list()
        for b in b_node:
            n = len(bags[b])
            if n == 2:  # 'b' represents an edge
                edges.append(b)
            elif n >= 3:  # 'b' represents a clique or a cycle
                strcs.append(b)
                weights.append(n * (n - 1) // 2 if x_bag[b] == 0 else n)

        if len(b_node) >= 3:  # intersection of at least 3 bags
            bags.append([node])
            x_bag.append(3)
            b = len(bags) - 1
            for e in edges:
                w_edge[(b, e)] = 1
            for s, w in zip(strcs, weights):
                w_edge[(b, s)] = w
        elif len(b_node) == 2:  # construct an edge between bags b1 and b2 to which the 'node' belongs
            b1, b2 = b_node[0], b_node[1]
            count = len(set(bags[b1]) & set(bags[b2]))
            w_edge[(b1, b2)] = max(count, w_edge.get((b1, b2), -1))

    # update the 'node2bag' maps
    node2bag = [[] for i in range(data.num_nodes)]
    for b in range(len(bags)):
        for node in bags[b]:
            node2bag[node].append(b)

    # construct the skeleton tree from the skeleton graph
    if len(w_edge) > 0:
        edge_index_T, weight = zip(*w_edge.items())
        edge_index = torch.tensor(edge_index_T).t()
        inv_weight = 50000 - torch.tensor(weight)
        graph = to_scipy_sparse_matrix(edge_index, inv_weight, len(bags))
        skeleton_tree = minimum_spanning_tree(graph)
        edge_index, _ = from_scipy_sparse_matrix(skeleton_tree)
        edge_index = to_undirected(edge_index, num_nodes=len(bags))
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)

    rows = [[i] * len(node2bag[i]) for i in range(data.num_nodes)]
    row = torch.tensor(list(chain.from_iterable(rows)))
    col = torch.tensor(list(chain.from_iterable(node2bag)))
    node2bag = torch.stack([row, col], dim=0).to(torch.long)

    return edge_index, node2bag, len(bags), torch.tensor(x_bag, dtype=torch.long), bags
