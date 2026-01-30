import os, json
import numpy as np
import pandas as pd
from functools import reduce

RAW_DIR = "AAA/florida-solar"
OUT_DIR = "Solar"
os.makedirs(OUT_DIR, exist_ok=True)

TIME_COL_CANDIDATES = ["time", "LocalTime"]
VALUE_COL_CANDIDATES = ["value", "Power(MW)"]

series_list = []
site_names = []

for fname in sorted(os.listdir(RAW_DIR)):
    if not fname.endswith(".csv"):
        continue
    
    path = os.path.join(RAW_DIR, fname)
    df = pd.read_csv(path)
    
    time_col = next(c for c in TIME_COL_CANDIDATES if c in df.columns)
    value_col = next(c for c in VALUE_COL_CANDIDATES if c in df.columns)
    
    df = df[[time_col, value_col]].copy()
    df[time_col] = pd.to_datetime(
        df[time_col],
        format="%m/%d/%y %H:%M",
        errors="raise"
    )
    df = df.sort_values(time_col)
    df = df.rename(columns={value_col: fname.replace(".csv", "")})
    df = df.set_index(time_col)
    series_list.append(df)
    site_names.append(fname.replace(".csv", ""))
    
merged = reduce(lambda left, right: left.join(right, how='inner'), series_list)
data = merged.values.astype(np.float32)

mean = data.mean(axis=0, keepdims=True)
std = data.std(axis=0, keepdims=True)
std[std == 0] = 1.0
data = (data - mean) / std

t1 = int(data.shape[0] * 0.7)
t2 = int(data.shape[0] * 0.9)
train = data[:t1]
val = data[t1:t2]
test = data[t2:]

np.save(os.path.join(OUT_DIR, "train.npy"), train)
np.save(os.path.join(OUT_DIR, "val.npy"), val)
np.save(os.path.join(OUT_DIR, "test.npy"), test)

meta = {
    "dataset": "Solar",
    "num_sites": len(site_names),
    "timestamps": merged.index.astype(str).tolist(),
    "sites": site_names,
    "normalization": "z-score",
    "split": "7:2:1",
    "shapes": {
        "train": list(train.shape),
        "val": list(val.shape),
        "test": list(test.shape)
    }
}

with open(os.path.join(OUT_DIR, "meta.json"), "w") as f:
    json.dump(meta, f, indent=2)
    
