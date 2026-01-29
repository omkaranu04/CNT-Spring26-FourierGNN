import os, time, argparse
import torch
import torch.nn as nn
import numpy as np
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
    'ECG':{'root_path':'./data/ECG', 'type':'0'},
    'Covid':{'root_path':'./data/Covid', 'type':'1'},
    'electricity':{'root_path':'./data/Electricity', 'type':'1'},
    'wiki':{'root_path':'./data/Wiki', 'type':'1'},
    'metr':{'root_path':'./data/METR-LA', 'type':'1'},
    'solar':{'root_path':'./data/Solar', 'type':'1'}
}

if args.data in data_parser.keys():
    data_info = data_parser[args.data]
    
data_dict = {
    'ECG': Dataset_ECG,
    'wiki': Dataset_Wiki,
    'solar': Dataset_Solar,
    'Covid': Dataset_DHFM,
    'electricity': Dataset_DHFM,
    'metr': Dataset_DHFM
}

Data = data_dict[args.data]
train_set = Data(data_info['root_path'], 'train', args.seq_length, args.pre_length, data_info['type'], args.train_ratio, args.val_ratio)
val_set = Data(data_info['root_path'], 'val', args.seq_length, args.pre_length, data_info['type'], args.train_ratio, args.val_ratio)
test_set = Data(data_info['root_path'], 'test', args.seq_length, args.pre_length, data_info['type'], args.train_ratio, args.val_ratio)

train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=False)
val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=False)
test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=0, drop_last=False)

device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
model = FGN(pre_length=args.pre_length, embed_size=args.embed_size, feature_size=args.feature_size, seq_length=args.seq_length, hidden_size=args.hidden_size)
model.to(device)
my_optim = torch.optim.RMSprop(params=model.parameters(), lr=args.learning_rate, eps=1e-08)
my_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=my_optim, gamma=args.decay_rate)
forecast_loss = nn.MSELoss(reduction='mean').to(device)


def validate(model, vali_loader):
    model.eval()
    cnt = 0
    loss_total = 0
    preds = []
    trues = []
    for i, (x, y) in enumerate(vali_loader):
        cnt += 1
        y = y.float().to("cuda:0")
        x = x.float().to("cuda:0")
        forecast = model(x)
        y = y.permute(0, 2, 1).contiguous()
        loss = forecast_loss(forecast, y)
        loss_total += float(loss)
        forecast = forecast.detach().cpu().numpy()  # .squeeze()
        y = y.detach().cpu().numpy()  # .squeeze()
        preds.append(forecast)
        trues.append(y)
    preds = np.concatenate(preds, axis=0)
    trues = np.concatenate(trues, axis=0)
    score = evaluate(trues, preds)
    print(f'RAW : MAPE {score[0]:7.9%}; MAE {score[1]:7.9f}; RMSE {score[2]:7.9f}.')
    model.train()
    return loss_total/cnt

def test():
    result_test_file = 'output/'+args.data+'/train'
    model = load_model(result_test_file, 48)
    model.eval()
    preds = []
    trues = []
    sne = []
    for index, (x, y) in enumerate(test_loader):
        y = y.float().to("cuda:0")
        x = x.float().to("cuda:0")
        forecast = model(x)
        y = y.permute(0, 2, 1).contiguous()
        forecast = forecast.detach().cpu().numpy()  # .squeeze()
        y = y.detach().cpu().numpy()  # .squeeze()
        preds.append(forecast)
        trues.append(y)

    preds = np.concatenate(preds, axis=0)
    trues = np.concatenate(trues, axis=0)
    score = evaluate(trues, preds)
    print(f'RAW : MAPE {score[0]:7.9%}; MAE {score[1]:7.9f}; RMSE {score[2]:7.9f}.')
    
if __name__ == '__main__':
    for epoch in range(args.train_epochs):
        epoch_start_time = time.time()
        model.train()
        loss_total = 0
        cnt = 0
        for index, (x, y) in enumerate(train_loader):
            cnt += 1
            y = y.float().to("cuda:0")
            x = x.float().to("cuda:0")
            forecast = model(x)
            y = y.permute(0, 2, 1).contiguous()
            loss = forecast_loss(forecast, y)
            loss.backward()
            my_optim.step()
            loss_total += float(loss)

        if (epoch + 1) % args.exponential_decay_step == 0:
            my_lr_scheduler.step()
        if (epoch + 1) % args.validate_freq == 0:
            val_loss = validate(model, val_loader)

        print('| end of epoch {:3d} | time: {:5.2f}s | train_total_loss {:5.4f} | val_loss {:5.4f}'.format(
                epoch, (time.time() - epoch_start_time), loss_total / cnt, val_loss))
        save_model(model, result_train_file, epoch)
        
        