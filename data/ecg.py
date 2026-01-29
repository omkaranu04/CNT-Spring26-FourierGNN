import os
import json
import numpy as np
import pandas as pd

train_path = "AAA/DownloadECG5000/ECG5000_TRAIN.txt"
test_path  = "AAA/DownloadECG5000/ECG5000_TEST.txt"

train_df = pd.read_csv(train_path, header=None, sep=r"\s+")
test_df  = pd.read_csv(test_path,  header=None, sep=r"\s+")

train_data = train_df.iloc[:, 1:].values
test_data  = test_df.iloc[:, 1:].values

all_data = np.vstack([train_data, test_data])

data = all_data.T.astype(np.float32)
mean = data.mean(axis=0, keepdims=True)
std  = data.std(axis=0, keepdims=True)
std[std == 0] = 1.0
data = (data - mean) / std

T = data.shape[0]
t_train = int(0.7 * T)
t_val   = int(0.9 * T)

train = data[:t_train]
val   = data[t_train:t_val]
test  = data[t_val:]

out_dir = "ECG"
os.makedirs(out_dir, exist_ok=True)

np.save(os.path.join(out_dir, "train.npy"), train)
np.save(os.path.join(out_dir, "val.npy"),   val)
np.save(os.path.join(out_dir, "test.npy"),  test)

meta = {
    "dataset": "ECG5000",
    "source": "UCR Time Series Archive",
    "original_shape": {
        "num_series": int(all_data.shape[0]),
        "time_steps": int(all_data.shape[1])
    },
    "final_shape": {
        "train": list(train.shape),
        "val": list(val.shape),
        "test": list(test.shape)
    },
    "normalization": "z-score",
    "split": "7:2:1"
}

with open(os.path.join(out_dir, "meta.json"), "w") as f:
    json.dump(meta, f, indent=2)