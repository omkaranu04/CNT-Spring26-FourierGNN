import os, json
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler

BASE_DIR = Path(__file__).resolve().parent
CSV_FILES = [BASE_DIR / "_RAW_DATASETS" / "WIKI" / "train_1.csv", BASE_DIR / "_RAW_DATASETS" / "WIKI" / "train_2.csv"]
OUT_DIR = BASE_DIR / "WIKI"
SAMPLE_K = 2000
RANDOM_SEED = 42
TRAIN_RATIO = 0.7
VAL_RATIO = 0.2

os.makedirs(OUT_DIR, exist_ok=True)
pd.options.mode.chained_assignment = None

dfs = []
for p in CSV_FILES:
    df = pd.read_csv(p)
    assert "Page" in df.columns, f"'Page' column not found in {p}"
    
    # Extract values (all columns except 'Page' column)
    # Matching: data[:, 1:data.shape[1]]
    raw_data = df.iloc[:, 1:].values
    
    # Create DataFrame and clean
    df_clean = pd.DataFrame(raw_data)
    dfs.append(df_clean)

# Combine all dataframes
combined = pd.concat(dfs, axis=0, ignore_index=True)

# Data cleaning: drop rows with any NaN values (matching Dataset_Wiki)
# Matching: df.dropna(axis=0, how='any')
combined = combined.dropna(axis=0, how='any')

# Sample if needed
num_pages = combined.shape[0]
if num_pages > SAMPLE_K:
    combined = combined.sample(n=SAMPLE_K, random_state=RANDOM_SEED)
    combined = combined.reset_index(drop=True)

# Transpose to match Dataset_Wiki structure
# Matching: .values.transpose()
data = combined.values.astype(np.float32).T
T, N = data.shape

print(f"Timesteps: {T}")
print(f"Number of pages (after cleaning & sampling): {N}")

# Split boundaries
train_end = int(T * TRAIN_RATIO)
val_end = int(T * (TRAIN_RATIO + VAL_RATIO))

# Fit MinMaxScaler ONLY on training data (matching Dataset_Wiki when type == '1')
mms = MinMaxScaler(feature_range=(0, 1))
mms.fit(data[:train_end])

# Transform entire dataset using training statistics
norm = mms.transform(data)

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
    "normalization": "min-max (fitted on train only, sklearn MinMaxScaler)",
    "data_cleaning": "dropped rows with any NaN values",
    "splits": {k: v.shape for k, v in splits.items()},
    "train_ratio": TRAIN_RATIO,
    "val_ratio": VAL_RATIO,
    "test_ratio": 1.0 - TRAIN_RATIO - VAL_RATIO,
    "sampled_pages": num_pages > SAMPLE_K,
    "sample_k": SAMPLE_K if num_pages > SAMPLE_K else num_pages,
    "random_seed": RANDOM_SEED
}

with open(os.path.join(OUT_DIR, "metadata.json"), "w") as f:
    json.dump(metadata, f, indent=2)
    
print("Preprocessing Complete")
