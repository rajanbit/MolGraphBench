# Importing modules
import torch
import numpy as np
from torch_geometric.data import Data
from torch_geometric.utils import dense_to_sparse
from torch_geometric.loader import DataLoader
from src.utils.calculateAdjacency import smiles2A
from src.utils.calculateEmbeddings import smiles2X

# Handling fingerprint array within DataLoaders
class MoleculeData(Data):
	def __cat_dim__(self, key, value, *args, **kwargs):
		if key == 'fp':
			return 0
		return super().__cat_dim__(key, value, *args, **kwargs)

# Dataloader function | GNN models
def loadData(data, batch_size=4, f_in=118, shuf=True):
	'''
	This function take numpy array of SMILES strings and target labels
	as input and create DataLoader object for GNN model input.

	Args:
		data: DataFrame
			data as pandas dataframe
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

	# Smiles array
	S = data["smiles"].to_numpy()
	# Fingerprint array
	FP = data.drop(["smiles", "target"], axis=1).to_numpy()
	# Target variable array
	y = data["target"].to_numpy()

	# Iterate over all data
	for i in range(len(data)):

		# Calculate adjacency matrix for each smiles
		A_matrix = smiles2A(S[i])

		# Calculate node embedding matrix for each smiles
		X_matrix = smiles2X(S[i])

		# Converting to tensor
		A_tensor = torch.tensor(A_matrix, dtype=torch.float)
		X_tensor = torch.tensor(X_matrix, dtype=torch.float)
		y_tensor = torch.tensor(y[i], dtype=torch.float)
		fp_tensor = torch.tensor(FP[i], dtype=torch.float).unsqueeze(0)

		# Sparse representation of adjacency matrix
		edge_index, edge_attr = dense_to_sparse(A_tensor)

		# Creating geometric data object using A, X, y, and FP
		data = MoleculeData(x=X_tensor, edge_index=edge_index, y=y_tensor, fp=fp_tensor)
		geodata.append(data)

	# Creating dataloader
	loader = DataLoader(geodata, batch_size=batch_size, shuffle=shuf)

	# Return loader
	return loader
