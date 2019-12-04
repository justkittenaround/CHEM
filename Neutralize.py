#code adapted from: https://www.rdkit.org/docs/Cookbook.html

from RD_load import smiles_data_loader
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np
import csv

tsv_path = '/home/ras/Desktop/Rachel/CHEM/hiv1_protease.tsv'

smiles_data = smiles_data_loader(tsv_path)


def _InitialiseNeutralisationReactions():
    patts= (
        # Imidazoles
        ('[n+;H]','n'),
        # Amines
        ('[N+;!H0]','N'),
        # Carboxylic acids and alcohols
        ('[$([O-]);!$([O-][#7])]','O'),
        # Thiols
        ('[S-;X1]','S'),
        # Sulfonamides
        ('[$([N-;X2]S(=O)=O)]','N'),
        # Enamines
        ('[$([N-;X2][C,N]=C)]','N'),
        # Tetrazoles
        ('[n-]','[nH]'),
        # Sulfoxides
        ('[$([S-]=O)]','S'),
        # Amides
        ('[$([N-]C=O)]','N'),
        )
    return [(Chem.MolFromSmarts(x),Chem.MolFromSmiles(y,False)) for x,y in patts]

_reactions=None
def NeutraliseCharges(smiles, reactions=None):
    global _reactions
    if reactions is None:
        if _reactions is None:
            _reactions=_InitialiseNeutralisationReactions()
        reactions=_reactions
    mol = Chem.MolFromSmiles(smiles)
    replaced = False
    for i,(reactant, product) in enumerate(reactions):
        while mol.HasSubstructMatch(reactant):
            replaced = True
            rms = AllChem.ReplaceSubstructs(mol, reactant, product)
            mol = rms[0]
    if replaced:
        return (Chem.MolToSmiles(mol,True), True)
    else:
        return (smiles, False)

def N_React(smiles_data):
    neutralized_bin = []
    smiles_bin = []
    gps_bin = []
    for idx, smile in enumerate(smiles_data):
        (molSmiles, neutralized) = NeutraliseCharges(smile)
        if neutralized:
            neutralized_bin.append(molSmiles)
            smiles_bin.append(smile)
            gps_bin.append(idx)
    neutralized_rxns = np.asarray(zip(gps_bin,smiles_bin, neutralized_bin))
    with open('neutralized_reactions.tsv', 'wt') as out_file:
        tsv_writer = csv.writer(out_file, delimiter='\t')
        for thing in neutralized_rxns:
            thingzero = thing[0]
            thingone = thing[1]
            thingtwo = thing[2]
            tsv_writer.writerow([thingzero, thingone, thingtwo])
    return neutralized_rxns












#
