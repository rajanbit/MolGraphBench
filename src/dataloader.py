# Importing modules
import torch
import numpy as np
from torch_geometric.data import Data
from torch_geometric.utils import dense_to_sparse
from torch_geometric.loader import DataLoader
from src.utils.calculateAdjacency import smiles2A
from src.utils.calculateEmbeddings import smiles2X

# Dataloader function | GNN models
def loadData(X, y, batch_size=4, f_in=118, shuf=True):
	'''
	This function take numpy array of SMILES strings and target labels
	as input and create DataLoader object for GNN model input.

	Args:
		X: numpy array
			Array of SMILES strings
		y: numpy array
			Array of target labels
		batch_size: int
			batch size for data loader
		f_in: int
			number of features (between 1 to 6)
		shuf: bool
			Shuffle dataset in DataLoader

	Return:
		DataLoader object
	'''

	geodata = []

	# Iterate over all smiles
	for i in range(len(X)):

		# Calculate adjacency matrix for each smiles
		A_matrix = smiles2A(X[i])

		# Calculate node embedding matrix for each smiles
		X_matrix = smiles2X(X[i])

		# Converting to tensor
		A_tensor = torch.tensor(A_matrix, dtype=torch.float)
		X_tensor = torch.tensor(X_matrix, dtype=torch.float)
		y_tensor = torch.tensor(y[i], dtype=torch.float)

		# Sparse representation of adjacency matrix
		edge_index, edge_attr = dense_to_sparse(A_tensor)

		# Creating geometric data object using A, X and y
		data = Data(x=X_tensor, edge_index=edge_index, y=y_tensor)
		geodata.append(data)
		
	# Creating dataloader
	loader = DataLoader(geodata, batch_size=batch_size, shuffle=shuf)

	# Return loader
	return loader
