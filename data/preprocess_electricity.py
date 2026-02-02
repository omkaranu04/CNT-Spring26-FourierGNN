import os, json
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler

BASE_DIR = Path(__file__).resolve().parent
TXT_PATH = BASE_DIR / "_RAW_DATASETS" / "ELECTRICITY" / "electricity.txt"
OUT_DIR = BASE_DIR / "ELECTRICITY"
TIME_COL = 0
TRAIN_RATIO = 0.7
VAL_RATIO = 0.2

os.makedirs(OUT_DIR, exist_ok=True)

df = pd.read_csv(
    TXT_PATH,
    sep=";",
    decimal=",",
    parse_dates=[TIME_COL]
)
df = df.sort_values(df.columns[TIME_COL])
df = df.drop(columns=[df.columns[TIME_COL]])
data = df.values.astype(np.float32)
T, N = data.shape

train_end = int(T * TRAIN_RATIO)
val_end = int(T * (TRAIN_RATIO + VAL_RATIO))

# Fit MinMaxScaler ONLY on training data (matching Dataset_ECG)
mms = MinMaxScaler(feature_range=(0, 1))
mms.fit(data[:train_end])

# Transform entire dataset using training statistics
norm_data = mms.transform(data)

train = norm_data[:train_end]
val = norm_data[train_end:val_end]
test = norm_data[val_end:]

for name, arr in tqdm(
    [("train", train), ("val", val), ("test", test)],
    desc="Saving Splits", leave=True):
    np.save(os.path.join(OUT_DIR, f"{name}.npy"), arr)
    
metadata = {
    "dataset": "Electricity",
    "num_timesteps": T,
    "num_meters": N,
    "granularity": "15 minutes",
    "normalization": "min-max (fitted on train only, sklearn MinMaxScaler)",
    "splits": {
        "train": train.shape,
        "val": val.shape,
        "test": test.shape
    },
    "train_ratio": TRAIN_RATIO,
    "val_ratio": VAL_RATIO,
    "test_ratio": 1.0 - TRAIN_RATIO - VAL_RATIO
}

with open(os.path.join(OUT_DIR, "metadata.json"), "w") as f:
    json.dump(metadata, f, indent=2)
    
print("Preprocessing Complete")

