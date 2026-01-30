import os, json, h5py
import numpy as np
from tqdm import tqdm
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
H5_PATH = BASE_DIR / "_RAW_DATASETS" / "METR-LA" / "METR-LA.h5"
DATASET_KEY = "df/block0_values"
OUT_DIR = BASE_DIR / "METR-LA"
TRAIN_RATIO = 0.7
VAL_RATIO = 0.2
EPS = 1e-8

os.makedirs(OUT_DIR, exist_ok=True)

print("Opening:", H5_PATH)
with h5py.File(H5_PATH, "r") as hf:
    def visitor(name, obj):
        typ = "Dataset" if isinstance(obj, h5py.Dataset) else "Group "
        shape = getattr(obj, "shape", None)
        if shape is not None:
            print(f"{typ:8} | {name:40} | shape={shape}")
        else:
            print(f"{typ:8} | {name:40}")
    hf.visititems(visitor)

def find_2d_dataset(hf):
    cand = []
    for k in hf.keys():
        try:
            d = hf[k]
            shape = getattr(d, "shape", None)
            if shape and len(shape) == 2:
                cand.append((k, shape[0] * shape[1]))
        except Exception:
            continue
    if not cand:
        return None
    cand.sort(key=lambda x: x[1], reverse=True)
    return cand[0][0]

with h5py.File(H5_PATH, "r") as hf:
    chosen_key = DATASET_KEY or find_2d_dataset(hf)
    if chosen_key is None:
        raise RuntimeError("No 2D dataset found in the .h5 file. Set DATASET_KEY manually.")
    ds = hf[chosen_key]
    data = np.array(ds, dtype=np.float32)
    
orig_shape = data.shape
rows, cols = orig_shape
if rows < cols and cols > 1000:
    data = data.T
    transposed = True
else:
    transposed = False
T, N = data.shape

mins = data.min(axis=0)
maxs = data.max(axis=0)
den = maxs - mins + EPS
norm_data = np.empty_like(data, dtype=np.float32)

for j in tqdm(range(N), desc="Normalizing Columns", leave=True):
    norm_data[:, j] = (data[:, j] - mins[j]) / den[j]
    
train_end = int(T * TRAIN_RATIO)
val_end = int(T * (TRAIN_RATIO + VAL_RATIO))

train = norm_data[:train_end]
val = norm_data[train_end:val_end]
test = norm_data[val_end:]

for name, arr in tqdm([("train", train), ("val", val), ("test", test)], desc="Saving splits"):
    np.save(os.path.join(OUT_DIR, f"{name}.npy"), arr)
    
metadata = {
    "h5_dataset_key": chosen_key,
    "original_shape": orig_shape,
    "processed_shape": norm_data.shape,
    "transposed_during_load": transposed,
    "num_timesteps": T,
    "num_nodes": N,
    "granularity": "5 minutes (METR-LA standard) -- verify if needed",
    "normalization": "min-max per variable (column) with epsilon",
    "min_per_column": mins.tolist(),
    "max_per_column": maxs.tolist(),
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
