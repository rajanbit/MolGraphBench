# Importing modules
import pandas as pd
from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize

# Function to preprocess SMILES
def preprocess(smiles_df):

	# Extract SMILES and target
	smiles_arr = smiles_df["smiles"].to_numpy()
	target_arr =  smiles_df["target"].to_numpy()

	# Var to store processed data
	p_smiles = []
	p_target = []

	for i in range(len(smiles_arr)):

		# SMILE and target
		smiles = smiles_arr[i]
		target = target_arr[i]

		# Convert to rdkit mol object
		mol = Chem.MolFromSmiles(smiles)

		if mol is not None:

			# Tautomer standardization
			tautomer_enumerator = rdMolStandardize.TautomerEnumerator()
			mol = tautomer_enumerator.Canonicalize(mol)

			# Neutralize charges
			uncharger = rdMolStandardize.Uncharger()
			mol = uncharger.uncharge(mol)

			# Remove counterions/salts
			largest = rdMolStandardize.LargestFragmentChooser()
			mol = largest.choose(mol)

			p_smiles.append(Chem.MolToSmiles(mol, canonical=True))
			p_target.append(target)
	# Return df with processed smiles and target
	return pd.DataFrame({"smiles":p_smiles, "target":p_target})
