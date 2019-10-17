from RD_load import smiles_data_loader
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np
import csv

tsv_path = '/home/ras/Desktop/Rachel/CHEM/hiv1_protease.tsv'
smiles_data = smiles_data_loader(tsv_path)


def _initialize_groups():

    fx_groups = ['C=O']

    return fx_groups


_groups = None
def deletions(smiles_data):
    _groups = initialize_groups()
    groups = _groups
    deletions_bin = []
    smiles_bin = []
    for m in smiles_data:
        mol = Chem.MolFromSmiles(m)
        for fx in fx_groups:
            try:
                matches = mol.GetSubstructMatches(Chem.MolFromSmarts(fx))
                if len(matches) > 0:
                    dmol = AllChem.DeleteSubstructs(mol,functional_group)
                    Chem.SanitizeMol(dmol)
                    smiles_bin.append(m)
                    deletions_bin.append(Chem.MolToSmiles(dmol))
            except:
                pass
    deletion_rxns = np.asarray(zip(smiles_bin, deletions_bin))
    with open('deletion_reactions.tsv', 'wt') as out_file:
        tsv_writer = csv.writer(out_file, delimiter='\t')
        for thing in deletion_rxns:
            thingone = thing[0]
            thingtwo = thing[1]
            tsv_writer.writerow([thingone, thingtwo])
    return deletion_rxns
