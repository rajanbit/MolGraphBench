# Importing modules
import torch
import torch.nn.functional as F
from torch.nn import Linear, Sequential, ReLU, Dropout
from torch_geometric.nn import GCNConv, GATConv, GINConv, SAGEConv, global_mean_pool

##################################### GNN MODEL ########################################
#########################################################################################

class GNNModel(torch.nn.Module):
	def __init__(self, num_features, hidden_dim, model_type='GCN', dropout=0.2):
		super().__init__()
		self.dropout_rate = dropout
		
		# Message passing layer
		if model_type == 'GCN': # GCN
			self.conv = GCNConv(num_features, hidden_dim)
		elif model_type == 'GAT': # GAT
			self.conv = GATConv(num_features, hidden_dim, heads=1, concat=False)
		elif model_type == 'GIN': # GIN
			gin_nn = Sequential(Linear(num_features, hidden_dim), ReLU(), Linear(hidden_dim, hidden_dim))
			self.conv = GINConv(gin_nn)
		elif model_type == 'GraphSAGE': # GraphSAGE
			self.conv = SAGEConv(num_features, hidden_dim, aggr='mean')

		# Finger-print layer
		self.fc_fp = Linear(1024, hidden_dim)


		# MLP regression head
		self.post_mp = Sequential(
			Linear(hidden_dim*2, hidden_dim), # Layer-1
			ReLU(),
			Dropout(p=self.dropout_rate),
			Linear(hidden_dim, 1) # Layer-2
		)

	def forward(self, data):
		x, fp, edge_index, batch = data.x, data.fp, data.edge_index, data.batch
		
		# Convolution
		x_graph = self.conv(x, edge_index)
		x_graph = F.relu(x_graph)
		
		# Readout
		x_graph = global_mean_pool(x_graph, batch)

		# FP regression
		x_fp = self.fc_fp(fp)
		x_fp = F.relu(x_fp)

		# Feature fusion
		x_combined = torch.cat([x_graph, x_fp], dim=1)

		# Regression
		return self.post_mp(x_combined)
