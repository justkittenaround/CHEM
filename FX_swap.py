from RD_load import smiles_data_loader
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np
import csv

tsv_path = '/home/ras/Desktop/Rachel/CHEM/hiv1_protease.tsv'
smiles_data = smiles_data_loader(tsv_path)



def _initialize_groups():
    fx_groups = np.asarray(['C=O', '[H+]', '[CX3]=[OX1]'])
    return fx_groups


_groups = None
def swap_fx(smiles_data):

    _groups = initialize_groups()
    groups = _groups
    swapped_bin = []
    smiles_bin = []
    for m in smiles_data:
        mol = Chem.MolFromSmiles(m)
        fx_bin = []
        for fx in groups:
            try:
                matches = mol.GetSubstructMatches(Chem.MolFromSmarts(fx))
                if len(matches) > 0:
                    fx_bin.append(Chem.MolFromSmarts(fx))
            except:
                pass
        if len(fx_bin) >= 2:
            fx_choice = np.random.choice(np.asarray(fx_bin), 2, replace=False)
            smol = AllChem.ReplaceSubstructs(mol, fx_choice[0], fx_choice[1])
            swapped_bin.append(Chem.MolToSmiles(smol[0]))
            smiles_bin.append(m)
            print('swapped')
        elif len(fx_bin) == 1:
            fx_poss = groups[groups != Chem.MolToSmiles(fx_bin[0])]
            swap_choice = np.random.choice(fx_poss, 1, replace=False)
            swap_choice = Chem.MolFromSmarts(swap_choice[0])
            smol = AllChem.ReplaceSubstructs(mol,fx_bin[0], swap_choice)
            swapped_bin.append(Chem.MolToSmiles(smol[0]))
            smiles_bin.append(m)
            print('swapped')
    swap_rxns = np.asarray(zip(smiles_bin, swapped_bin))
    with open('deletion_reactions.tsv', 'wt') as out_file:
        tsv_writer = csv.writer(out_file, delimiter='\t')
        for thing in deletion_rxns:
            thingone = thing[0]
            thingtwo = thing[1]
            tsv_writer.writerow([thingone, thingtwo])
    return sawp_rxns





###
