import h5py
import os, json
import numpy as np
import pandas as pd

os.makedirs('METR-LA', exist_ok=True)

with h5py.File("AAA/METR-LA.h5", "r") as f:
    arr = f["df"]["block0_values"][:]
    
if arr.shape[0] < arr.shape[1]:
    arr = arr.T
    
t1 = int(arr.shape[0] * 0.7)
t2 = int(arr.shape[0] * 0.9)

np.save("METR-LA/train.npy", arr[:t1])
np.save("METR-LA/val.npy", arr[t1:t2])
np.save("METR-LA/test.npy", arr[t2:])

meta = {
    "split": "7:2:1",
    "shape": {
        "train": list(arr[:t1].shape),
        "val": list(arr[t1:t2].shape),
        "test": list(arr[t2:].shape)
    }
}

with open(os.path.join("METR-LA", "meta.json"), "w") as f:
    json.dump(meta, f, indent=2)
    
