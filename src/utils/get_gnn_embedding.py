# Importing modules
import torch
import numpy as np

# Model testing function | GNN models
def GetGNNEmbed(model, testing_loader):

	'''
	This function take trained model and test dataloader, to generate
	GNN based embeddings

	Args:
		model:  GNN model
			Pytorch model object

		testing_loader: Validation/Testing data loader object
				Geometric DataLoader object

	Return:
		list for GNN embeddings
	'''

	model.eval()
	gnn_embeddings = []
	y_test = []

	# Device
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	
	# Load model to device
	model = model.to(device)

	with torch.no_grad():

		# Validation/Testing loop
		for data in testing_loader:
			data = data.to(device)
			out, embed = model(data)
			gnn_embeddings.append(embed)
			y_test.append(data.y.view(-1).cpu())

	# vstack the embedding output
	gnn_embeddings = np.vstack(gnn_embeddings)
	y_test = torch.cat(y_test).tolist()

	# Return embeddings
	return gnn_embeddings, y_test
