import os, json
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
CSV_PATH = BASE_DIR / "_RAW_DATASETS" / "TRAFFIC" / "traffic.csv"
OUT_DIR = BASE_DIR / "TRAFFIC"
TIME__COL = "date"
TRAIN_RATIO = 0.7
VAL_RATIO = 0.2
EPS = 1e-8

os.makedirs(OUT_DIR, exist_ok=True)

df = pd.read_csv(CSV_PATH)
df[TIME__COL] = pd.to_datetime(df[TIME__COL])
df = df.sort_values(TIME__COL)
df = df.drop(columns=[TIME__COL])
data = df.values.astype(np.float32)
T, N = data.shape

mins = data.min(axis=0)
maxs = data.max(axis=0)
noram_data = np.empty_like(data)
for i in tqdm(range(N), desc="Normalizing", leave=True):
    noram_data[:, i] = (data[:, i] - mins[i]) / (maxs[i] - mins[i] + EPS)
    
train_end = int(T * TRAIN_RATIO)
val_end = int(T * (TRAIN_RATIO + VAL_RATIO))

train = noram_data[:train_end]
val = noram_data[train_end:val_end]
test = noram_data[val_end:]

for name, arr in tqdm(
    [("train", train), ("val", val), ("test", test)],
    desc="Saving Splits", leave=True):
    np.save(os.path.join(OUT_DIR, f"{name}.npy"), arr)
    
metadata = {
    "dataset": "Traffic",
    "num_timesteps": T,
    "num_sensors": N,
    "granularity": "1 hour",
    "normalization": "min-max per sensor",
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