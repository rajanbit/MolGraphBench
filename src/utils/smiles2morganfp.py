# Importing modules
import numpy as np
import pandas as pd
from rdkit import DataStructs
from rdkit import Chem
from rdkit.Chem.rdMolDescriptors import GetMorganFingerprintAsBitVect

# Function to convert smiles into morgan fingerprint (FP)
def smiles2MorganFP(smiles, radius=2, nBits=2048):
	mol = Chem.MolFromSmiles(smiles)
	if mol is None:
		return None
	fp = GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=nBits)
	arr = np.zeros((nBits,), dtype=int)
	DataStructs.ConvertToNumpyArray(fp, arr)
	return arr

# Generate morgan FP for list/array of smiles
def MorganFP(smiles_list):
	# Generate FP
	fps = []
	valid_smiles = []
	for smi in smiles_list:
		fp = smiles2MorganFP(smi)
		if fp is not None:
			fps.append(fp)
			valid_smiles.append(smi)

	# FP dataframe
	fp_df = pd.DataFrame(fps, index=valid_smiles, 
				columns=[f"FP_{i}" for i in range(len(fps[0]))])
	return fp_df
