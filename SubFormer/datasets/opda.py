import numpy as np
from typing import Callable, Optional
from rdkit.Chem.rdchem import BondType as BT
from rdkit.Chem.rdchem import HybridizationType

import torch
from tqdm import tqdm

from torch_geometric.data import (
    Data,
    InMemoryDataset,
    download_url,
    extract_zip,
)

from torch_geometric.utils import one_hot
import pandas as pd
import sqlite3
import ast

from rdkit import Chem

import struct


def decode_positions(byte_data):
    try:
        n = len(byte_data) // 8
        return struct.unpack('d' * n, byte_data)
    except:
        return None


def get_mol(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return mol


class OPDADataset(InMemoryDataset):

    def __init__(self, root: str, transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None,
                 pre_filter: Optional[Callable] = None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['opda.db']

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        pass

    def process(self):

        try:
            conn = sqlite3.connect(self.raw_paths[0])
            print(self.raw_paths[0])
        except Exception as e:
            print(e)

        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        table_names = cursor.fetchall()
        print(f"Table Names : {table_names}")

        if table_names:
            table_name = table_names[0][0]
            df = pd.read_sql_query(f'SELECT * FROM {table_name}', conn)
            print(df.head())

        conn.close()

        df['key_value_pairs'] = df['key_value_pairs'].apply(ast.literal_eval)
        df['positions_decoded'] = df['positions'].apply(decode_positions)
        for key in df['key_value_pairs'].iloc[0]:
            df[key] = df['key_value_pairs'].apply(lambda x: x.get(key))

        types = {
            'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5, 'C': 6, 'N': 7, 'O': 8,
            'F': 9, 'Ne': 10, 'Na': 11, 'Mg': 12, 'Al': 13, 'Si': 14, 'P': 15, 'S': 16,
            'Cl': 17, 'K': 19, 'Ar': 18, 'Ca': 20, 'Sc': 21, 'Ti': 22, 'V': 23, 'Cr': 24,
            'Mn': 25, 'Fe': 26, 'Ni': 28, 'Co': 27, 'Cu': 29, 'Zn': 30, 'Ga': 31, 'Ge': 32,
            'As': 33, 'Se': 34, 'Br': 35, 'Kr': 36, 'Rb': 37, 'Sr': 38, 'Y': 39, 'Zr': 40,
            'Nb': 41, 'Mo': 42, 'Tc': 43, 'Ru': 44, 'Rh': 45, 'Pd': 46, 'Ag': 47, 'Cd': 48,
            'In': 49, 'Sn': 50, 'Sb': 51, 'Te': 52, 'I': 53, 'Xe': 54, 'Cs': 55, 'Ba': 56,
            'La': 57, 'Ce': 58, 'Pr': 59, 'Nd': 60, 'Pm': 61, 'Sm': 62, 'Eu': 63, 'Gd': 64,
            'Tb': 65, 'Dy': 66, 'Ho': 67, 'Er': 68, 'Tm': 69, 'Yb': 70, 'Lu': 71, 'Hf': 72,
            'Ta': 73, 'W': 74, 'Re': 75, 'Os': 76, 'Ir': 77, 'Pt': 78, 'Au': 79, 'Hg': 80,
            'Tl': 81, 'Pb': 82, 'Bi': 83, 'Th': 90, 'Pa': 91, 'U': 92
        }
        bonds = {BT.SINGLE: 0, BT.DOUBLE: 1, BT.TRIPLE: 2, BT.AROMATIC: 3}

        data_list = []
        for i in tqdm(range(5366)):
            mol = get_mol(df['SMILES'][i])
            try:
                mol = Chem.AddHs(mol)
            except:
                continue

            target = [df['KS_gap'][i], df['rho'][i], df['E_homo'][i], df['E_lumo'][i], df['dip'][i]]
            pos = np.array(df['positions_decoded'][i]).reshape(-1, 3)

            conf = Chem.Conformer(mol.GetNumAtoms())
            for i in range(pos.shape[0]):
                conf.SetAtomPosition(i, (pos[i, 0], pos[i, 1], pos[i, 2]))

            mol.AddConformer(conf)

            mol_without_h = Chem.RemoveHs(mol)
            conf = mol_without_h.GetConformer()
            pos = np.array([list(conf.GetAtomPosition(i)) for i in range(mol_without_h.GetNumAtoms())])
            pos = torch.tensor(pos).float()
            mol = mol_without_h
            N = mol_without_h.GetNumAtoms()
            type_idx = []
            atomic_number = []
            aromatic = []
            sp = []
            sp2 = []
            sp3 = []
            for atom in mol_without_h.GetAtoms():
                type_idx.append(types[atom.GetSymbol()])
                atomic_number.append(atom.GetAtomicNum())
                aromatic.append(1 if atom.GetIsAromatic() else 0)
                hybridization = atom.GetHybridization()
                sp.append(1 if hybridization == HybridizationType.SP else 0)
                sp2.append(1 if hybridization == HybridizationType.SP2 else 0)
                sp3.append(1 if hybridization == HybridizationType.SP3 else 0)

            z = torch.tensor(atomic_number, dtype=torch.long)

            row, col, edge_type = [], [], []
            for bond in mol_without_h.GetBonds():
                start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                row += [start, end]
                col += [end, start]
                edge_type += 2 * [bonds[bond.GetBondType()]]

            edge_index = torch.tensor([row, col], dtype=torch.long)
            edge_type = torch.tensor(edge_type, dtype=torch.long)
            edge_attr = one_hot(edge_type, num_classes=len(bonds))

            perm = (edge_index[0] * N + edge_index[1]).argsort()
            edge_index = edge_index[:, perm]
            edge_type = edge_type[perm]
            edge_attr = edge_attr[perm]

            x = torch.tensor(type_idx).long()
            assert x.size(0) == pos.size(0)
            y = torch.tensor(target).float()
            smiles = Chem.MolToSmiles(mol, isomericSmiles=True)
            data = Data(
                x=x,
                z=z,
                pos=pos,
                edge_index=edge_index,
                smiles=smiles,
                edge_attr=edge_type,
                y=y,
                idx=i,
                mol=mol_without_h,
            )

            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)

            data_list.append(data)

        torch.save(self.collate(data_list), self.processed_paths[0])
