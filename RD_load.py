import numpy as np
import pandas as pd
from rdkit import Chem

tsv_path = '/home/ras/Desktop/Rachel/CHEM/hiv1_protease.tsv'

def data_loader(tsv_path):
    data = np.asarray(pd.read_csv(tsv_path, sep="\t"))
    return data

def smiles_data_loader(tsv_path):
    data = np.asarray(pd.read_csv(tsv_path, sep="\t"))
    smiles = data[:, 1]
    return smiles

def rd_data_loader(tsv_path):
    data = np.asarray(pd.read_csv(tsv_path, sep="\t"))
    smiles = data[:, 1]
    rd_data = []
    for smile in smiles:
        m = Chem.MolFromSmiles(smile)
        rd_data.append(m)
    return rd_data
