import os, time, argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from utils.utils import save_model, load_model, evaluate
from data.a_data_loader import Dataset_DHFM, Dataset_ECG, Dataset_Solar, Dataset_Wiki
from model.FourierGNN import FGN

parser = argparse.ArgumentParser(description='fourier graph network for multivariate time series forecasting')
parser.add_argument('--data', type=str, default='ECG', help='data set')
parser.add_argument('--feature_size', type=int, default='140', help='feature size')
parser.add_argument('--seq_length', type=int, default=12, help='inout length')
parser.add_argument('--pre_length', type=int, default=12, help='predict length')
parser.add_argument('--embed_size', type=int, default=128, help='hidden dimensions')
parser.add_argument('--hidden_size', type=int, default=256, help='hidden dimensions')
parser.add_argument('--train_epochs', type=int, default=100, help='train epochs')
parser.add_argument('--batch_size', type=int, default=32, help='input data batch size')
parser.add_argument('--learning_rate', type=float, default=0.00001, help='optimizer learning rate')
parser.add_argument('--exponential_decay_step', type=int, default=5)
parser.add_argument('--validate_freq', type=int, default=1)
parser.add_argument('--early_stop', type=bool, default=False)
parser.add_argument('--decay_rate', type=float, default=0.5)
parser.add_argument('--train_ratio', type=float, default=0.7)
parser.add_argument('--val_ratio', type=float, default=0.2)
parser.add_argument('--device', type=str, default='cuda:0', help='device')

args = parser.parse_args()
print(f"Training Configuration: {args}")

result_train_file = os.path.join('output', args.data, 'train')
result_test_file = os.path.join('output', args.data, 'test')
if not os.path.exists(result_train_file):
    os.makedirs(result_train_file)
if not os.path.exists(result_test_file):
    os.makedirs(result_test_file)
    
data_parser = {
    ''
}

