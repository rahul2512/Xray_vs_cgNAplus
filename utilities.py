import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import Parallel, delayed
import re

def load_raw_data(filename):
    cols = [f"x{i}" for i in range(1, 31)] + ["pdb_id", "hexamer"]
    # sed -i '' '/,/!d' jm*
    # sed -i '' 's/, /;/g'  jm*
    # sed -i '' 's/ /;/g'  jm*
    df = pd.read_csv(f"data/{filename}.txt",sep=";", header=None, names=cols)
    df = df[~((df.pdb_id.isna()) | (df.hexamer.isna()))]
    df["tetramer"] = df["hexamer"].str[1:-1]
    df["dimer"] = df["hexamer"].str[2:-2]
    return df

