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


# Unified Dataset class for all datasets
class TimeSeriesDataset(Dataset):
    """
    Unified dataset class for all time series datasets (TRAFFIC, ECG, COVID, 
    ELECTRICITY, METR-LA, SOLAR, WIKI).
    """
    def __init__(self, root_path, flag, seq_len, pre_len, type):
        """
        Args:
            root_path: Path to the dataset folder (e.g., 'data/WIKI/')
            flag: 'train', 'val', or 'test'
            seq_len: Length of input sequence
            pre_len: Length of prediction horizon
            type: '1' for MinMaxScaler normalization, else no scaling
            train_ratio: (unused, kept for backward compatibility)
            val_ratio: (unused, kept for backward compatibility)
        """
        assert flag in ["train", "val", "test"]

        self.flag = flag
        self.seq_len = seq_len
        self.pre_len = pre_len

        data = load_split(root_path, flag)
        self.data = data

    def __getitem__(self, idx):
        x = self.data[idx : idx + self.seq_len]
        y = self.data[idx + self.seq_len : idx + self.seq_len + self.pre_len]
        return x, y

    def __len__(self):
        return len(self.data) - self.seq_len - self.pre_len
