# Importing modules
import torch
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU, Dropout
from torch_geometric.nn import GCNConv, GATv2Conv, GINConv, SAGEConv
from torch_geometric.nn import GraphNorm, global_mean_pool, global_add_pool

##################################### GNN MODEL ########################################
#########################################################################################

class GNNModel(torch.nn.Module):
	def __init__(self, num_features, hidden_dim, model_type='GCN', num_layers=3, dropout=0.2):
		super().__init__()
		self.model_type = model_type
		self.dropout_rate = dropout # Fix: Define dropout_rate

		self.convs = torch.nn.ModuleList()
		self.batch_norms = torch.nn.ModuleList()

		# Project input features to hidden_dim once at the start
		self.input_proj = Linear(num_features, hidden_dim)

		for i in range(num_layers):
			if model_type == 'GCN':
				conv = GCNConv(hidden_dim, hidden_dim, add_self_loops=True)
			elif model_type == 'GAT':
				# heads=4 with concat=True results in hidden_dim output
				conv = GATv2Conv(hidden_dim, hidden_dim // 4, heads=4, concat=True)
			elif model_type == 'GIN':
				nn = Sequential(
					Linear(hidden_dim, hidden_dim), 
					ReLU(), 
					Linear(hidden_dim, hidden_dim)
				)
				conv = GINConv(nn, train_eps=True)
			elif model_type == 'GraphSAGE':
				conv = SAGEConv(hidden_dim, hidden_dim, aggr='mean', normalize=True)

			self.convs.append(conv)
			self.batch_norms.append(GraphNorm(hidden_dim))

		# Finger-print projection
		self.fp_proj = Linear(1024, hidden_dim)

		self.post_mp = Sequential(
			Linear(hidden_dim*2, hidden_dim),
			ReLU(),
			Dropout(p=self.dropout_rate),
			Linear(hidden_dim, 1)
		)

	def forward(self, data):
		x, fp, edge_index, batch = data.x, data.fp, data.edge_index, data.batch

		# Initial projection to hidden space
		x = F.relu(self.input_proj(x))

		# Message Passing Loop
		for i, conv in enumerate(self.convs):
			identity = x # Save for residual connection

			x = conv(x, edge_index)
			x = self.batch_norms[i](x)
			x = F.relu(x)

			# Residual connection (ensures deep models actually train)
			x = x + identity
			x = F.dropout(x, p=self.dropout_rate, training=self.training)

		# Global Pooling (Graph-level representation)
		if self.model_type == 'GIN':
			x_graph = global_add_pool(x, batch)
		else:
			x_graph = global_mean_pool(x, batch)

		# FP regression
		x_fp = self.fp_proj(fp)
		x_fp = F.relu(x_fp)

		# Feature fusion
		x_combined = torch.cat([x_graph, x_fp], dim=1)

		# Final Regression Head
		return self.post_mp(x_combined), None

