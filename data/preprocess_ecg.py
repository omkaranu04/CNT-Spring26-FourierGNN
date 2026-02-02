import os, json
import numpy as np
from tqdm import tqdm
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler

BASE_DIR = Path(__file__).resolve().parent
TRAIN_TXT = BASE_DIR / "_RAW_DATASETS" / "ECG" / "ECG5000_TRAIN.txt"
TEST_TXT = BASE_DIR / "_RAW_DATASETS" / "ECG" / "ECG5000_TEST.txt"
OUT_DIR = BASE_DIR / "ECG"
TRAIN_RATIO = 0.7
VAL_RATIO = 0.2

os.makedirs(OUT_DIR, exist_ok=True)

def load_txt(path):
    data = np.loadtxt(path)
    return data[:, 1:]  # Remove label column

train_raw = load_txt(TRAIN_TXT)
test_raw = load_txt(TEST_TXT)

# Concatenate all data
data = np.concatenate([train_raw, test_raw], axis=0).astype(np.float32)
T, L = data.shape

# Determine split boundaries
train_end = int(T * TRAIN_RATIO)
val_end = int(T * (TRAIN_RATIO + VAL_RATIO))

# Fit scaler ONLY on training data (matching Dataset_ECG)
mms = MinMaxScaler(feature_range=(0, 1))
mms.fit(data[:train_end])  # Fit only on training portion

# Transform entire dataset using training statistics
data = mms.transform(data)

# Split the normalized data
splits = {
    "train": data[:train_end],
    "val": data[train_end:val_end],
    "test": data[val_end:]
}

for k in tqdm(splits, desc="Saving Splits", leave=True):
    np.save(os.path.join(OUT_DIR, f"{k}.npy"), splits[k])
    
metadata = {
    "dataset": "ECG5000",
    "num_samples": T,
    "signal_length": L,
    "normalization": "min-max (fitted on train only)",
    "splits": {
        "train": splits["train"].shape,
        "val":   splits["val"].shape,
        "test":  splits["test"].shape
    },
    "train_ratio": TRAIN_RATIO,
    "val_ratio": VAL_RATIO,
    "test_ratio": 1.0 - TRAIN_RATIO - VAL_RATIO
}

with open(os.path.join(OUT_DIR, "metadata.json"), "w") as f:
    json.dump(metadata, f, indent=2)
    
print("Preprocessing complete.")
