import torch
from rdkit import Chem
from rdkit.Chem.rdchem import BondType
from torch_geometric.data import Data
from torch_geometric.transforms import Compose, VirtualNode
from torch_geometric.utils import tree_decomposition
from SubFormer.data.utils import comp_pe, comp_deg

bonds = [BondType.SINGLE, BondType.DOUBLE, BondType.TRIPLE, BondType.AROMATIC]

pe_dim = 32


def get_transform(add_virtual_node=False, pedim=10):
    global pe_dim
    pe_dim = pedim
    if add_virtual_node:
        return Compose([OGBTransform(), JunctionTree(), VirtualNode()])
    else:
        return Compose([OGBTransform(), JunctionTree()])


def get_transform_zinc(add_virtual_node=False, pedim=10):
    global pe_dim
    pe_dim = pedim
    if add_virtual_node:
        return Compose([JunctionTree(), VirtualNode()])
    else:
        return Compose([JunctionTree()])

def get_transform_opda(add_virtual_node=False, pedim=10):
    global pe_dim
    pe_dim = pedim
    if add_virtual_node:
        return Compose([OPDATransform(), JunctionTree(), VirtualNode()])
    else:
        return Compose([OPDATransform(), JunctionTree()])


def mol_from_data(data):
    mol = Chem.RWMol()
    x = data.x if data.x.dim() == 1 else data.x[:, 0]
    for z in x.tolist():
        mol.AddAtom(Chem.Atom(z))

    row, col = data.edge_index
    mask = row < col
    row, col = row[mask].tolist(), col[mask].tolist()

    bond_type = data.edge_attr
    bond_type = bond_type if bond_type.dim() == 1 else bond_type[:, 0]
    bond_type = bond_type[mask].tolist()

    for i, j, bond in zip(row, col, bond_type):
        bond = int(bond)
        assert 1 <= bond <= 4
        mol.AddBond(i, j, bonds[bond - 1])

    return mol.GetMol()


class JunctionTreeData(Data):
    def __inc__(self, key, item, *args):
        if key == 'tree_edge_index':
            return self.x_clique.size(0)
        elif key == 'atom2clique_index':
            return torch.tensor([[self.x.size(0)], [self.x_clique.size(0)]])
        else:
            return super(JunctionTreeData, self).__inc__(key, item, *args)


class OGBTransform(object):
    # OGB saves atom and bond types zero-index based. We need to revert that.
    def __call__(self, data: Data) -> Data:
        data.x[:, 0] += 1
        data.edge_attr[:, 0] += 1
        return data

class OPDATransform(object):
    def __call__(self, data: Data) -> Data:
        # print(data.edge_attr.shape)
        data.edge_attr[:] += 1
        return data

class JunctionTree(object):
    def __call__(self, data: Data):
        mol = mol_from_data(data)
        tree = tree_decomposition(mol, return_vocab=True)
        tree_edge_index, atom2clique_index, num_cliques, x_clique = tree
        data = JunctionTreeData(**{k: v for k, v in data})

        data.tree_edge_index = tree_edge_index
        data.atom2clique_index = atom2clique_index
        data.num_cliques = num_cliques
        data.x_clique = x_clique
        data.edge_attr_graph = data.edge_attr
        data.edge_index_graph = data.edge_index

        ## graph level
        data.graph_degree = comp_deg(data.edge_index_graph, data.num_nodes)
        data.graph_lpe, data.graph_lpeval = comp_pe(
            dim=pe_dim,
            edge_attr=None,
            edge_index=data.edge_index_graph,
            num_nodes=data.num_nodes,
            normalization='rw',
            use_edge_attr=False,
        )

        ## tree level
        data.tree_degree = comp_deg(tree_edge_index, num_cliques)
        data.tree_lpe, data.tree_lpeval = comp_pe(
            dim=pe_dim,
            edge_attr=None,
            edge_index=tree_edge_index,
            num_nodes=num_cliques,
            normalization='rw',
            use_edge_attr=False,
        )

        return data
