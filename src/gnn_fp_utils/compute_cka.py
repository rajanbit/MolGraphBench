# Importing modules
import torch
import numpy as np
from scipy.spatial.distance import pdist, squareform

def centering(K):
	n = K.shape[0]
	unit = np.ones([n, n])
	I = np.eye(n)
	H = I - unit / n
	return np.dot(np.dot(H, K), H)

def rbf_kernel(X, sigma=None):
	dists = squareform(pdist(X, 'sqeuclidean'))
	if sigma is None:
		sigma = np.median(dists) # Median trick for bandwidth
	K = np.exp(-dists / (2 * sigma))
	return K

# Function to compute CKA score
def compute_cka(feat1, feat2):
	K = rbf_kernel(feat1)
	L = rbf_kernel(feat2)
	
	K_centered = centering(K)
	L_centered = centering(L)
	
	# CKA Formula: HSIC(K, L) / sqrt(HSIC(K, K) * HSIC(L, L))
	hsic_kl = np.sum(K_centered * L_centered)
	hsic_kk = np.sum(K_centered * K_centered)
	hsic_ll = np.sum(L_centered * L_centered)
	
	cka_score = hsic_kl / np.sqrt(hsic_kk * hsic_ll)
	
	return cka_score

