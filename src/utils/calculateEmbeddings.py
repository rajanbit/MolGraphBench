# Importing modules
import numpy as np
from rdkit import Chem

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

	# Node embedding matrix
	X = []

	# Node i embedding
	for node in mol.GetAtoms():
		node_X = [
		node.GetAtomicNum(), node.GetTotalDegree(),
		node.GetValence(Chem.ValenceType.IMPLICIT), node.GetFormalCharge(),
		int(node.GetHybridization()), int(node.GetIsAromatic())]
		
		# Append node i embedding in X
		X.append(node_X)

	# Converting to numpy array
	X = np.array(X, dtype=np.float32)

	# Return H
	return X

