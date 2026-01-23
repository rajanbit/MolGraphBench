# Importing modules
import torch
from torch.optim import Adam
from torch.nn.functional import mse_loss
import warnings
warnings.filterwarnings(
    "ignore",
    message="Attempting to run cuBLAS"
)

# Model training function
def TrainGNN(model, training_loader, epochs=10, learning_rate=0.01):

	'''
	This function take model, dataloader, number of epochs, learning rate 
	as input and train the model till n epochs using dataset from dataloader

	Args:
		model:  GNN model
			Pytorch model object

		training_loader: Training data loader object
				Geometric DataLoader object

		epochs: Number of epochs
			Int object

		learning_rate: Learning rate
			Float object

	Return:
		Trained model
	'''

	# Optimizer
	optimizer = Adam(model.parameters(), lr=learning_rate)

	# Device
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	
	# Load model to device
	model = model.to(device)

	# Training loop
	for e in range(epochs):
		for data in training_loader:
			data = data.to(device)
			optimizer.zero_grad()
			out = model(data.x, data.edge_index, data.batch)
			loss = mse_loss(out.view(-1).cpu(), data.y.view(-1).cpu())
			loss.backward()
			optimizer.step()

	# Return trained model
	return model

# Model testing function
def TestGNN(model, testing_loader):

	'''
	This function take trained model and test dataloader, to perform
	prediction for evaluating the model's performance on test set

	Args:
		model:  GNN model
			Pytorch model object

		testing_loader: Validation/Testing data loader object
				Geometric DataLoader object

	Return:
		list for y_test and y_pred
	'''

	model.eval()
	y_pred, y_test = [], []

	# Device
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	
	# Load model to device
	model = model.to(device)

	with torch.no_grad():

		# Validation/Testing loop
		for data in testing_loader:
			data = data.to(device)
			out = model(data.x, data.edge_index, data.batch)
			out = out.view(-1).cpu()
			y = data.y.view(-1).cpu()
			y_pred.append(out)
			y_test.append(y)

	# Flattening tensor
	y_test = torch.cat(y_test).tolist()
	y_pred = torch.cat(y_pred).tolist()

	# Return y_test and y_pred
	return y_test, y_pred

