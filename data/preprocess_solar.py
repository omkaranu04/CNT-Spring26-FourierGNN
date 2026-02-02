import os, json, glob
import numpy as np
import pandas as pd
import datetime
from tqdm import tqdm
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "_RAW_DATASETS" / "SOLAR"
OUT_DIR = BASE_DIR / "SOLAR"
TRAIN_RATIO = 0.7
VAL_RATIO = 0.2
DATE_FORMAT = "%m/%d/%y %H:%M"

os.makedirs(OUT_DIR, exist_ok=True)

# Get all files starting with 'DA_' (matching Dataset_Solar logic)
csv_files = sorted(glob.glob(os.path.join(DATA_DIR, "DA_*.csv")))
assert len(csv_files) > 0, "No DA_*.csv files found in the dataset directory."

solar_data = []
time_data = []

for path in tqdm(csv_files, desc="Loading CSV files", leave=True):
    data = pd.read_csv(path).values
    
    # Extract time column (first column)
    raw_time = data[:, 0:1]
    if len(time_data) == 0:
        time_data = raw_time
    
    # Extract data columns (all except first)
    raw_data = data[:, 1:data.shape[1]]
    raw_data = raw_data.transpose()  # Transpose as in Dataset_Solar
    solar_data.append(raw_data)

# Combine all solar data
solar_data = np.array(solar_data).squeeze(1).transpose()
time_data = np.array(time_data)

# Concatenate time and solar data
out = np.concatenate((time_data, solar_data), axis=1)

# Filter for daytime hours (8 AM to 5 PM) as in Dataset_Solar
filtered_data = []
for item in out:
    tmp = item[0]
    dt = datetime.datetime.strptime(tmp, DATE_FORMAT)
    if dt.hour >= 8 and dt.hour <= 17:
        # Exclude last column (out.shape[1]-1) as in Dataset_Solar
        filtered_data.append(item[1:out.shape[1]-1])

data = np.array(filtered_data, dtype=np.float32)
T, N = data.shape

print(f"Filtered timesteps (8AM-5PM): {T}")
print(f"Number of features: {N}")

# Split boundaries
train_end = int(T * TRAIN_RATIO)
val_end = int(T * (TRAIN_RATIO + VAL_RATIO))

# Fit MinMaxScaler ONLY on training data (matching Dataset_Solar when type == '1')
mms = MinMaxScaler(feature_range=(0, 1))
mms.fit(data[:train_end])

# Transform entire dataset using training statistics
norm_data = mms.transform(data)

# Create splits
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
    "num_features": N,
    "num_files_processed": len(csv_files),
    "granularity": "10 minutes (filtered to 8AM-5PM)",
    "time_filter": "8AM to 5PM only",
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

