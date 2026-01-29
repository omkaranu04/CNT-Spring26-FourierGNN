import os, json
import numpy as np
import pandas as pd

os.makedirs('Covid', exist_ok=True)

df = pd.read_csv("AAA/time_series_covid19_confirmed_US.csv")
date_cols = df.columns[df.columns.str.match(r"\d+/\d+/\d+")].tolist()

state_time_series = (
    df.groupby("Province_State")[date_cols]
    .sum()
)
states = state_time_series.index.tolist()
dates = date_cols

data = state_time_series.values.T.astype(np.float32)
mean = data.mean(0, keepdims=True)
std = data.std(0, keepdims=True)
std[std == 0] = 1.0
data = (data - mean) / std

T = data.shape[0]
t1 = int(0.6 * T)
t2 = int(0.8 * T)

train = data[:t1]
val = data[t1:t2]
test = data[t2:]

np.save("Covid/train.npy", train)
np.save("Covid/val.npy", val)
np.save("Covid/test.npy", test)

meta = {
    "states": states,
    "dates": dates,
    "normalization": {
        "type": "z-score",
        "mean": mean.flatten().tolist(),
        "std": std.flatten().tolist()
    },
    "split": "6:2:2",
    "shape": {
        "train": list(train.shape),
        "val": list(val.shape),
        "test": list(test.shape)
    }
}

with open(os.path.join("Covid", "meta.json"), "w") as f:
    json.dump(meta, f, indent=2)
    