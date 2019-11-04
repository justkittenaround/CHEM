#read in molecule from smiles to smiles
from rdkit import Chem
molecule = Chem.MolFromSmiles('c1ccncc1')
smiles = Chem.MolToSmiles(molecule)

#show molecule as image
from rdkit.Chem import Draw
from PIL import Image
im = Draw.MolToImage(molecule)
im.show()


from rdkit.Chem import AllChem
from rdkit.Chem.Draw import MolDrawing, DrawingOptions
# combo = Chem.CombineMols(molecule,fragment)
combo = Chem.MolFromSmiles(m)
edcombo = Chem.EditableMol(combo)
DrawingOptions.includeAtomNumbers=True
edcombo.AddBond(5,10,order=Chem.rdchem.BondType.SINGLE)



check_og = []
for idx, smile in enumerate(smiles_data):
    if '.' in smile:
        check_og.append(idx)

fix = []
for neut in check_neut:
     if not neut in check_og:
             fix.append(neut)
