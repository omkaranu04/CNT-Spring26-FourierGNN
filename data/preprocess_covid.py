import os, json
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
CSV_PATH = BASE_DIR / "_RAW_DATASETS" / "COVID" / "covid.csv"
OUT_DIR = BASE_DIR / "COVID"
DATE_COL = "date"
TRAIN_RATIO = 0.6
VAL_RATIO = 0.2

os.makedirs(OUT_DIR, exist_ok=True)

df = pd.read_csv(CSV_PATH)
df[DATE_COL] = pd.to_datetime(df[DATE_COL])
df = df.sort_values(DATE_COL)

ts_cols = df.select_dtypes(include=[np.number]).columns.tolist()
df = df[ts_cols]

data = df.values.astype(np.float32)

T = data.shape[0]
train_end = int(T * TRAIN_RATIO)
val_end = int(T * (TRAIN_RATIO + VAL_RATIO))

train_data = data[:train_end]

mins = train_data.min(axis=0)
maxs = train_data.max(axis=0)

data = (data - mins) / (maxs - mins + 1e-8)

splits = {
    "train": data[:train_end],
    "val": data[train_end:val_end],
    "test": data[val_end:],
}

for k in tqdm(splits, desc="Saving Splits", leave=True):
    np.save(os.path.join(OUT_DIR, f"{k}.npy"), splits[k])

metadata = {
    "dataset": "COVID-19 California Hospitalizations",
    "num_timesteps": T,
    "num_nodes": data.shape[1],
    "granularity": "1 day",
    "normalization": "min-max (train-only, per variable)",
    "splits": {
        "train": splits["train"].shape,
        "val": splits["val"].shape,
        "test": splits["test"].shape
    },
    "train_ratio": TRAIN_RATIO,
    "val_ratio": VAL_RATIO,
    "test_ratio": 1.0 - TRAIN_RATIO - VAL_RATIO,
    "columns": ts_cols
}

with open(os.path.join(OUT_DIR, "metadata.json"), "w") as f:
    json.dump(metadata, f, indent=2)

print("Preprocessing Complete")
