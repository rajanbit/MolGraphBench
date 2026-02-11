# Importing modules
import os
import pandas as pd
from urllib.request import urlretrieve

# Download function 1
def download_esol():
	URL = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/delaney-processed.csv"
	FilePath = "data/ESOL.csv"

	try:
		urlretrieve(URL, FilePath)

	except Exception as e:
		print(f"An unexpected error occurred while downloading: {e}")

# Download function 2
def download_lipophilicity():
	URL = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/Lipophilicity.csv"
	FilePath = "data/Lipophilicity.csv"

	try:
		urlretrieve(URL, FilePath)

	except Exception as e:
		print(f"An unexpected error occurred while downloading: {e}")

# Download function 3
def download_rt():

	files = ["HILIC-train.txt","HILIC-val.txt","HILIC-test.txt"]
	url_path = "https://raw.githubusercontent.com/Qiong-Yang/GNN-TL/main/data/"
	dfs = []

	for f in files:
		try:
			urlretrieve(url_path+f, 'data/'+f)
			dfs.append(pd.read_csv('data/'+f, sep="\t"))
			os.remove('data/'+f)

		except Exception as e:
			print(f"An unexpected error occurred while downloading: {e}")

	final_df = pd.concat(dfs)
	final_df.to_csv("data/RT.csv", index=False, quoting=False)

# Download function 4
def download_b3db():
	URL = "https://raw.githubusercontent.com/theochem/B3DB/main/B3DB/B3DB_regression.tsv"
	FilePath = "data/B3DB.tsv"

	try:
		urlretrieve(URL, FilePath)

	except Exception as e:
		print(f"An unexpected error occurred while downloading: {e}")

# Download all datasets
def download_all():
	download_esol()
	download_lipophilicity()
	download_rt()
	download_b3db()

