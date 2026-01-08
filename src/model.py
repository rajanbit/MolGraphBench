# Importing modules
import torch
import torch.nn as nn
from torch.nn.functional import relu, elu
from torch_geometric.nn import GCNConv, GATConv, GINConv, SAGEConv
from torch_geometric.nn import global_mean_pool
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet

################################ MACHINE LEARNING MODELS ################################
#########################################################################################

# ML Models With Hyperparameters
ML_Models = {
"LR": LinearRegression(fit_intercept=True, positive=False), # ML Model 1
"ENet": ElasticNet(alpha=0.01, l1_ratio=0.2, max_iter=1000,random_state=42), # ML Model 2
"SVM": SVR(C=10, epsilon=0.01, gamma="scale", kernel="rbf"), # ML Model 3
"RF": RandomForestRegressor(n_estimators=100, max_depth=None, max_features="sqrt", # ML Model 4
				min_samples_leaf=1, min_samples_split=2, random_state=42)
}

################################ DEEP LEARNING MODELS ###################################
#########################################################################################

# GNN Model 1
class SimpleGCN(torch.nn.Module):
	def __init__(self, num_features, hidden_dim, num_classes):
		super().__init__()
		self.conv = GCNConv(num_features, hidden_dim) # GCN layer
		self.fc = torch.nn.Linear(hidden_dim, num_classes) # FC(fully connected/dense) layer 

	def forward(self, x, edge_index, batch):
		x = self.conv(x, edge_index) # GCN Conv
		x = relu(x) # ReLU
		x = global_mean_pool(x, batch) # Global pooling
		x = self.fc(x) # FC regression
		return x

# GNN Model 2
class SimpleGAT(torch.nn.Module):
	def __init__(self, num_features, hidden_dim, num_classes):
		super().__init__()
		# GAT layer
		self.conv = GATConv(num_features, hidden_dim, 
					heads=8, # Single attention head
					concat = False,
					dropout=0.6, # Dropout from original paper
					negative_slope=0.2) # LeakyReLU original paper
		# FC(fully connected/dense) layer 
		self.fc = torch.nn.Linear(hidden_dim, num_classes)

	def forward(self, x, edge_index, batch):
		x = self.conv(x, edge_index) # GCN Conv
		x = relu(x) # ReLU
		x = global_mean_pool(x, batch) # Global pooling
		x = self.fc(x) # FC regression
		return x

# GNN Model 3
class SimpleGIN(torch.nn.Module):
	def __init__(self, num_features, hidden_dim, num_classes):
		super().__init__()
	        # MLP for GIN
		self.mlp = nn.Linear(num_features, hidden_dim)

		# GIN layer
		self.conv = GINConv(nn=self.mlp, train_eps=False)
		# FC(fully connected/dense) layer 
		self.fc = torch.nn.Linear(hidden_dim, num_classes)

	def forward(self, x, edge_index, batch):
		x = self.conv(x, edge_index) # GCN Conv
		x = relu(x) # ReLU
		x = global_mean_pool(x, batch) # Global pooling
		x = self.fc(x) # FC regression
		return x

# GNN Model 4
class SimpleGraphSAGE(nn.Module):
	def __init__(self, num_features, hidden_dim, num_classes):
		super().__init__()
		# GraphSAGE Layer
		self.conv = SAGEConv(num_features, hidden_dim, aggr="mean")
		# FC(fully connected/dense) layer 
		self.fc = nn.Linear(hidden_dim, num_classes)

	def forward(self, x, edge_index, batch):
		x = self.conv(x, edge_index) # GCN Conv
		x = relu(x) # ReLU
		x = global_mean_pool(x, batch) # Global pooling
		x = self.fc(x) # FC regression
		return x

