# Importing modules
import numpy as np
from rdkit import Chem

# Function for one hot encodding
def one_hot_encoding(value, choices):
	encoding = [0] * len(choices)
	try:
		index = choices.index(value)
		encoding[index] = 1
	except ValueError:
		encoding[-1] = 1
	return encoding

# Function to calculate node embedding matrix from smiles
def smiles2X(smiles):

	"""
	This function take smiles and calculate node embedding matrix 
	of molecules where N (nodes) are atoms and E (edges) are bonds
	
	Args:
		smiles: str
			Molecule as SMILES string

	Return:
		Node embedding matrix (X)
	"""

	# Converting smiles to rdkit mol object
	mol = Chem.MolFromSmiles(smiles)
	if mol is None: return None
	
	# Define categorical ranges
	elements = ['H', 'C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 
				'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 
				'Pd', 'Co', 'Se', 'Ti', 'Zn', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 
				'In', 'Mn', 'Zr', 'Cr', 'Pt', 'Hg', 'Pb', 'Unknown']

	# Node embedding matrix
	X = []

	# Node i embedding
	for node in mol.GetAtoms():
		node_X = (
			one_hot_encoding(node.GetSymbol(), elements) +
			one_hot_encoding(node.GetTotalDegree(), list(range(11))) +
			one_hot_encoding(node.GetTotalNumHs(), list(range(11))) +
			one_hot_encoding(node.GetValence(Chem.ValenceType.IMPLICIT), list(range(11))) +
			one_hot_encoding(node.GetTotalValence(), list(range(11))) +
			one_hot_encoding(node.GetFormalCharge(), list(range(-5, 6))) +
			one_hot_encoding(node.GetHybridization(), [
				Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
				Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.SP3D,
				Chem.rdchem.HybridizationType.SP3D2, Chem.rdchem.HybridizationType.OTHER]) +
			one_hot_encoding(node.GetNumRadicalElectrons(), list(range(6))) +
			one_hot_encoding(node.GetChiralTag(), [
				Chem.rdchem.ChiralType.CHI_UNSPECIFIED, Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
				Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW, Chem.rdchem.ChiralType.CHI_OTHER,
				'Unknown']) +
			[float(node.GetIsAromatic())] +
			[float(node.IsInRing())]
		)
		X.append(node_X)

	return np.array(X, dtype=np.float32)