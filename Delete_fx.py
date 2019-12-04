from RD_load import smiles_data_loader
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np
import csv

tsv_path = 'hiv1_protease.tsv'
smiles_data = smiles_data_loader(tsv_path)


def _initialize_groups():
    fx_groups = (['C=O', '[H+]', '[CX3]=[OX1]', '[CX3](=[OX1])C', '[OX1]=CN', '[CX3](=[OX1])O', '[CX3](=[OX1])[F,Cl,Br,I]', '[CX3H1](=O)[#6]', '[CX3](=[OX1])[OX2][CX3](=[OX1])', '[NX3][CX3](=[OX1])[#6]', '[NX3][CX3]=[NX3+]', '[NX3,NX4+][CX3](=[OX1])[OX2,OX1-]',  '[CX3](=O)[OX2H1]', '[NX3][CX2]#[NX1]', '[#6][CX3](=O)[#6]', '[OD2]([#6])[#6]', '[NX3;H2,H1;!$(NC=O)]', '[NX3,NX4+][CX4H]([*])[CX3](=[OX1])[O,N]', "[$(*-[NX2-]-[NX2+]#[NX1]),$(*-[NX2]=[NX2+]=[NX1-])]", "[NX2]=N", '[NX3][NX3]', '[NX3][NX2]=[*]', '[CX3;$([C]([#6])[#6]),$([CH][#6])]=[NX2][#6]', '[NX3+]=[CX3]', '[CX3](=[OX1])[NX3H][CX3](=[OX1])', '[$([NX3](=[OX1])(=[OX1])O),$([NX3+]([OX1-])(=[OX1])O)]', '[NX1]#[CX2]', '[$([NX3](=O)=O),$([NX3+](=O)[O-])][!#8]', '[NX2]=[OX1]', '[$([#7+][OX1-]),$([#7v5]=[OX1]);!$([#7](~[O])~[O]);!$([#7]=[#7])]', '[OX2H]', '[$([OH]-*=[!#6])]', '[OX2,OX1-][OX2,OX1-]', '[$(P(=[OX1])([$([OX2H]),$([OX1-]),$([OX2]P)])([$([OX2H]),$([OX1-]),$([OX2]P)])[$([OX2H]),$([OX1-]),$([OX2]P)]),$([P+]([OX1-])([$([OX2H]),$([OX1-]),$([OX2]P)])([$([OX2H]),$([OX1-]),$([OX2]P)])[$([OX2H]),$([OX1-]),$([OX2]P)])]', '[#16X2H]', '[#16X2H0]', '[$([#16X3](=[OX1])[OX2H0]),$([#16X3+]([OX1-])[OX2H0])]', '[$([#16X4](=[OX1])=[OX1]),$([#16X4+2]([OX1-])[OX1-])]', '[$([#16X3]=[OX1]),$([#16X3+][OX1-])]', '[$([#16X4](=[OX1])(=[OX1])([OX2H,OX1H0-])[OX2][#6]),$([#16X4+2]([OX1-])([OX1-])([OX2H,OX1H0-])[OX2][#6])]', '[$([#16X4]([NX3])(=[OX1])(=[OX1])[OX2][#6]),$([#16X4+2]([NX3])([OX1-])([OX1-])[OX2][#6])]', '[#16X2][OX2H,OX1H0-]', '[#16X2][OX2H0]', '[CX3](=[OX1])[F,Cl,Br,I]'])
    return fx_groups


_groups = None
def deletions(smiles_data):
    _groups = _initialize_groups()
    groups = _groups
    deletions_bin = []
    smiles_bin = []
    gps_bin = []
    for idx, m in enumerate(smiles_data):
        mol = Chem.MolFromSmiles(m)
        pregps = []
        pres = []
        predels = []
        for fx in fx_groups:
            try:
                matches = mol.GetSubstructMatches(Chem.MolFromSmarts(fx))
                if len(matches) > 0:
                    dmol = AllChem.DeleteSubstructs(mol,Chem.MolFromSmarts(fx))
                    Chem.SanitizeMol(dmol)
                    pres.append(m)
                    pregps.append(idx)
                    predels.append(Chem.MolToSmiles(dmol))
            except:
                pass
        if len(predels) > 0:
            num = int(np.random.randint(0, len(predels), 1))
            deletions_bin.append(predels[num])
            smiles_bin.append(pres[num])
            gps_bin.append(pregps[num])
    deletion_rxns = np.asarray(zip(gps_bin, smiles_bin, deletions_bin))
    with open('deletion_reactions.tsv', 'wt') as out_file:
        tsv_writer = csv.writer(out_file, delimiter='\t')
        for thing in deletion_rxns:
            thingzero = thing[0]
            thingone = thing[1]
            thingtwo = thing[2]
            tsv_writer.writerow([thingzero, thingone, thingtwo])
    return deletion_rxns

fx_groups = _initialize_groups()
d = deletions(smiles_data)
#https://www.daylight.com/dayhtml_tutorials/languages/smarts/smarts_examples.html
