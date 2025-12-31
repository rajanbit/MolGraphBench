# Importing modules
import numpy as np
from rdkit import Chem

# Function to calculate adjacency matrix from smiles
def smiles2A(smiles, weighted = True):

	'''
	This function take smiles and calculate adjacency matrix
	of molecules where N (nodes) are atoms and E (edges) are bonds

	Args:
		smiles: str
			Molecule as SMILES string
		weighted: bool
			Indicates whether weighted adjacency matrix to be calculated.

	Return:
		Adjacency matrix (A)
	'''

	# Converting smiles to rdkit mol object
	mol = Chem.MolFromSmiles(smiles)
	
	# Weighted adjacency matrix with 0 in diagonal
	A = Chem.GetAdjacencyMatrix(mol, useBO=weighted)
	
	# Identity matrix 
	I = np.identity(A.shape[0])
	
	# Final adjacency matrix
	A = A + I

	# Return A
	return A

