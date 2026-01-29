import os
import datetime
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, Dataloader
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

# Traffic Dataset Class
class Dataset_DHFM(Dataset):
    def __init__(self, root_path, flag, seq_len, pre_len, type, train_ratio, val_ratio):
        