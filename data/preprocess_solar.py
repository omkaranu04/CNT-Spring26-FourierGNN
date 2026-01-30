import os, json, glob
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "_RAW_DATASETS" / "SOLAR"
OUT_DIR = BASE_DIR / "SOLAR"
TRAIN_RATIO = 0.7
VAL_RATIO = 0.2
EPS = 1e-8
TIME_COL = "LocalTime"
VALUE_COL = "Power(MW)"
DATE_FORMAT = "%m/%d/%y %H:%M"

os.makedirs(OUT_DIR, exist_ok=True)

csv_files = sorted(glob.glob(os.path.join(DATA_DIR, "*.csv")))
assert len(csv_files) > 0, "No CSV files found in the dataset directory."

dfs = []
for path in tqdm(csv_files, desc="Loading CSV files", leave=True):
    df = pd.read_csv(path)

    df[TIME_COL] = pd.to_datetime(df[TIME_COL], format=DATE_FORMAT)
    df = df.sort_values(TIME_COL)
    df = df.set_index(TIME_COL)

    # Keep only the value column
    dfs.append(df[[VALUE_COL]])
    
merged = pd.concat(dfs, axis=1, join="inner")
merged.columns = list(range(len(dfs)))
data = merged.values.astype(np.float32)
timesteps = merged.index.values
T, N = data.shape

print(f"Aligned timesteps: {T}")
print(f"Number of sites: {N}")
print(f"Time range: {merged.index[0]} → {merged.index[-1]}")

mins = data.min(axis=0)
maxs = data.max(axis=0)
norm_data = np.empty_like(data)
for i in tqdm(range(N), desc="Normalizing", leave=True):
    norm_data[:, i] = (data[:, i] - mins[i]) / (maxs[i] - mins[i] + EPS)
    
train_end = int(T * TRAIN_RATIO)
val_end = int(T * (TRAIN_RATIO + VAL_RATIO))

train = norm_data[:train_end]
val = norm_data[train_end:val_end]
test = norm_data[val_end:]

for name, arr in tqdm(
    [("train", train), ("val", val), ("test", test)],
    desc="Saving Splits", leave=True):
    np.save(os.path.join(OUT_DIR, f"{name}.npy"), arr)
    
metadata = {
    "dataset": "Solar",
    "num_timesteps": T,
    "num_sites": N,
    "granularity": "5 minutes",
    "normalization": "min-max per site",
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