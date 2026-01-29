import os, json
import numpy as np
import pandas as pd

os.makedirs('Wiki', exist_ok=True)

df = pd.read_csv("AAA/wiki_train_1.csv", index_col=0)
arr = df.values.astype(float).T

mean = arr.mean(0, keepdims=True)
std = arr.std(0, keepdims=True)
std[std == 0] = 1

arr = (arr - mean) / std

T = arr.shape[0]
t1 = int(T * 0.7)
t2 = int(T * 0.9)

np.save("Wiki/train.npy", arr[:t1])
np.save("Wiki/val.npy", arr[t1:t2])
np.save("Wiki/test.npy", arr[t2:])
json.dump({
    "cols": df.index.tolist(),
    "mean": mean.flatten().tolist(),
    "std": std.flatten().tolist()
}, open("Wiki/meta.json", "w"), indent=2
)