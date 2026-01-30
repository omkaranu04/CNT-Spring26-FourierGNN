import os, json
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
CSV_FILES = [BASE_DIR / "_RAW_DATASETS" / "WIKI" / "train_1.csv", BASE_DIR / "_RAW_DATASETS" / "WIKI" / "train_2.csv"]
OUT_DIR = BASE_DIR / "WIKI"
SAMPLE_K = 2000
RANDOM_SEED = 42
TRAIN_RATIO = 0.7
VAL_RATIO = 0.2
EPS = 1e-8

os.makedirs(OUT_DIR, exist_ok=True)
pd.options.mode.chained_assignment = None

dfs = []
for p in CSV_FILES:
    df = pd.read_csv(p)
    assert "Page" in df.columns, f"'Page' column not found in {p}"
    df = df.set_index("Page")
    dfs.append(df)
    
combined = pd.concat(dfs, axis=0)
combined = combined[~combined.index.duplicated(keep='first')]
combined = combined.apply(pd.to_numeric, errors='coerce').fillna(0.0)
num_pages = combined.shape[0]
if num_pages > SAMPLE_K:
    combined = combined.sample(n=SAMPLE_K, random_state=RANDOM_SEED)
    combined = combined.sort_index()
data = combined.values.astype(np.float32).T
T, N = data.shape

mins = data.min(axis=0)
maxs = data.max(axis=0)
den = maxs - mins + EPS
norm = np.empty_like(data, dtype=np.float32)
for j in tqdm(range(N), desc="Normalizing", leave=True):
    norm[:, j] = (data[:, j] - mins[j]) / den[j]
    
train_end = int(T * TRAIN_RATIO)
val_end = int(T * (TRAIN_RATIO + VAL_RATIO))

splits = {
    "train": norm[:train_end],
    "val":   norm[train_end:val_end],
    "test":  norm[val_end:]
}
for name, arr in tqdm(
    splits.items(),
    desc="Saving Splits", leave=True):
    np.save(os.path.join(OUT_DIR, f"{name}.npy"), arr)
    
metadata = {
    "dataset": "Wiki (Wikipedia daily views)",
    "num_timesteps": T,
    "num_pages": N,
    "date_range_columns": combined.columns[0] + " ... " + combined.columns[-1],
    "normalization": "min-max per page (column) with eps",
    "splits": {k: v.shape for k, v in splits.items()},
    "train_ratio": TRAIN_RATIO,
    "val_ratio": VAL_RATIO,
    "test_ratio": 1.0 - TRAIN_RATIO - VAL_RATIO,
    "sampled_pages": num_pages > SAMPLE_K,
    "sample_k": SAMPLE_K if num_pages > SAMPLE_K else num_pages,
    "random_seed": RANDOM_SEED,
    "page_index": combined.index.tolist()
}

with open(os.path.join(OUT_DIR, "metadata.json"), "w") as f:
    json.dump(metadata, f, indent=2)
    
print("Preprocessing Complete")