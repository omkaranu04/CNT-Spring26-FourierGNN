import os
import numpy as np
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler

# Helper: load split safely
def load_split(root_path, flag):
    fname = {
        "train": "train.npy",
        "val": "val.npy",
        "test": "test.npy"
    }[flag]
    path = os.path.join(root_path, fname)
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found")
    return np.load(path)


# Traffic Dataset (formerly DHFM)
class Dataset_DHFM(Dataset):
    def __init__(self, root_path, flag, seq_len, pre_len, type, train_ratio, val_ratio):
        assert flag in ["train", "val", "test"]

        self.flag = flag
        self.seq_len = seq_len
        self.pre_len = pre_len

        data = load_split(root_path, flag)

        # optional scaling (kept for compatibility)
        if type == "1":
            mms = MinMaxScaler()
            data = mms.fit_transform(data)

        self.data = data

    def __getitem__(self, idx):
        x = self.data[idx : idx + self.seq_len]
        y = self.data[idx + self.seq_len : idx + self.seq_len + self.pre_len]
        return x, y

    def __len__(self):
        return len(self.data) - self.seq_len - self.pre_len


# ECG / COVID / Electricity / METR
class Dataset_ECG(Dataset):
    def __init__(self, root_path, flag, seq_len, pre_len, type, train_ratio, val_ratio):
        assert flag in ["train", "val", "test"]

        self.flag = flag
        self.seq_len = seq_len
        self.pre_len = pre_len

        data = load_split(root_path, flag)

        # optional scaling (kept)
        if type == "1":
            mms = MinMaxScaler()
            data = mms.fit_transform(data)

        self.data = data

    def __getitem__(self, idx):
        x = self.data[idx : idx + self.seq_len]
        y = self.data[idx + self.seq_len : idx + self.seq_len + self.pre_len]
        return x, y

    def __len__(self):
        return len(self.data) - self.seq_len - self.pre_len


# Solar Dataset
class Dataset_Solar(Dataset):
    def __init__(self, root_path, flag, seq_len, pre_len, type, train_ratio, val_ratio):
        assert flag in ["train", "val", "test"]

        self.flag = flag
        self.seq_len = seq_len
        self.pre_len = pre_len

        data = load_split(root_path, flag)

        # optional scaling
        if type == "1":
            mms = MinMaxScaler()
            data = mms.fit_transform(data)

        self.data = data

    def __getitem__(self, idx):
        x = self.data[idx : idx + self.seq_len]
        y = self.data[idx + self.seq_len : idx + self.seq_len + self.pre_len]
        return x, y

    def __len__(self):
        return len(self.data) - self.seq_len - self.pre_len


# Wiki Dataset
class Dataset_Wiki(Dataset):
    def __init__(self, root_path, flag, seq_len, pre_len, type, train_ratio, val_ratio):
        assert flag in ["train", "val", "test"]

        self.flag = flag
        self.seq_len = seq_len
        self.pre_len = pre_len

        data = load_split(root_path, flag)

        # optional scaling
        if type == "1":
            mms = MinMaxScaler()
            data = mms.fit_transform(data)

        self.data = data

    def __getitem__(self, idx):
        x = self.data[idx : idx + self.seq_len]
        y = self.data[idx + self.seq_len : idx + self.seq_len + self.pre_len]
        return x, y

    def __len__(self):
        return len(self.data) - self.seq_len - self.pre_len