# Importing modules
import torch
from torch.optim import Adam
from torch.nn.functional import mse_loss
import warnings
warnings.filterwarnings(
	"ignore",
	message="Attempting to run cuBLAS"
)

# Model training function | GNN models
def TrainGNN(model, training_loader, validation_loader, epochs=10, learning_rate=0.01, w_decay=1e-4):

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
	optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=w_decay)

	# Device
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	
	# Load model to device
	model = model.to(device)

	# Early stopping setup
	patience = 10
	best_val_loss = float('inf')
	epochs_without_improvement = 0

	# Training loop
	for e in range(epochs):
		train_loss = 0
		for data in training_loader:
			data = data.to(device)
			optimizer.zero_grad()

############################### CKA Analysis Block ##########################

#			out, _ = model(data)

#############################################################################

			out = model(data)
			loss = mse_loss(out.view(-1), data.y.view(-1))
			loss.backward()
			optimizer.step()
			train_loss += loss.item()


		# Validation loss
		model.eval()
		val_loss = 0
		with torch.no_grad():
			for data in validation_loader:
				data = data.to(device)
				out = model(data)
				loss = mse_loss(out.view(-1), data.y.view(-1))
				val_loss += loss.item()

		avg_train_loss = train_loss / len(training_loader)
		avg_val_loss = val_loss / len(validation_loader)

		# Early stopping
		if avg_val_loss < best_val_loss:
			best_val_loss = avg_val_loss
			epochs_without_improvement = 0
			best_model_weights = model.state_dict().copy()

		else:
			epochs_without_improvement += 1

		if epochs_without_improvement >= patience:
			model.load_state_dict(best_model_weights)
			break


	# Return trained model
	return model

# Model testing function | GNN models
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
			out = model(data)
			y_pred.append(out.view(-1).cpu())
			y_test.append(data.y.view(-1).cpu())

	# Flattening tensor
	y_test = torch.cat(y_test).tolist()
	y_pred = torch.cat(y_pred).tolist()

	# Return y_test and y_pred
	return y_test, y_pred
