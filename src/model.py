# Importing modules
import torch
import torch.nn as nn
from torch.nn.functional import relu, elu
from torch_geometric.nn import GCNConv, GATConv, GINConv, SAGEConv
from torch_geometric.nn import global_mean_pool

# Model 1
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

# Model 2
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

# Model 3
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

# Model 4
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
