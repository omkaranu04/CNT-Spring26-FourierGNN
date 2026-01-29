import numpy as np
import pandas as pd
import os, json

os.makedirs('Electricity', exist_ok=True)
df = pd.read_csv("AAA/DownloadElectricity/electricity.txt", sep=";", decimal=",", header=0, index_col=0, low_memory=False)
arr = df.values.astype(float)

if arr.shape[0] < arr.shape[1]:
    arr = arr.T
    
t1 = int(arr.shape[0] * 0.7)
t2 = int(arr.shape[0] * 0.9)

np.save("Electricity/train.npy", arr[:t1])
np.save("Electricity/val.npy", arr[t1:t2])
np.save("Electricity/test.npy", arr[t2:])

meta = {
    "split": "7:2:1",
    "shape": {
        "train": list(arr[:t1].shape),
        "val": list(arr[t1:t2].shape),
        "test": list(arr[t2:].shape)
    }
}
with open(os.path.join("Electricity", "meta.json"), "w") as f:
    json.dump(meta, f, indent=2)
    