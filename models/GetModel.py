import argparse
import os
import torch
import random
import numpy as np
import sys
sys.path.append("..")
import Informer, Autoformer, Transformer, DLinear, Linear, NLinear,hqhGNN,GNNLinear,CrossTwo,CrossFFTGNN
fix_seed = 2021
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)
# Informer=y[3,:,4]+3
# Autoformer=y[4,:,5]+3.5
# Fedformer=y[5,:,6]+1.3
parser = argparse.ArgumentParser(description='Autoformer & Transformer family for Time Series Forecasting')

# basic config
parser.add_argument('--is_training', type=int, default=1, help='status')
parser.add_argument('--train_only', type=bool, required=False, default=False, help='perform training on full input dataset without validation and testing')
parser.add_argument('--model_id', type=str, default='test', help='model id')
parser.add_argument('--model', type=str, default='Autoformer',
                    help='model name, options: [Autoformer, Informer, Transformer]')

# data loader
parser.add_argument('--data', type=str, default='ETTh2', help='dataset type')
parser.add_argument('--root_path', type=str, default='/home/hqh/Transformer/LTSF-Linear-main/dataset', help='root path of the data file')
parser.add_argument('--data_path', type=str, default='ETTh2.csv', help='data file')
parser.add_argument('--features', type=str, default='M', # feature是预测模式，默认是多变量预测多变量
                    help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
parser.add_argument('--freq', type=str, default='h',
                    help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

# forecasting task
parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
parser.add_argument('--label_len', type=int, default=48, help='start token length')
parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')


# DLinear
parser.add_argument('--individual', action='store_true', default=False, help='DLinear: a linear layer for each variate(channel) individually')
# Formers 
parser.add_argument('--embed_type', type=int, default=0, help='0: default 1: value embedding + temporal embedding + positional embedding 2: value embedding + temporal embedding 3: value embedding + positional embedding 4: value embedding')
parser.add_argument('--enc_in', type=int, default=7, help='encoder input size') # DLinear with --individual, use this hyperparameter as the number of channels
parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
parser.add_argument('--c_out', type=int, default=7, help='output size')
parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
parser.add_argument('--e_layers', type=int, default=1, help='num of encoder layers')
parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
parser.add_argument('--factor', type=int, default=1, help='attn factor')
parser.add_argument('--distil', action='store_false',
                    help='whether to use distilling in encoder, using this argument means not using distilling',
                    default=True)
parser.add_argument('--dropout', type=float, default=0.05, help='dropout')
parser.add_argument('--embed', type=str, default='timeF',
                    help='time features encoding, options:[timeF, fixed, learned]')
parser.add_argument('--activation', type=str, default='gelu', help='activation')
parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
parser.add_argument('--do_predict', action='store_true', help='whether to predict unseen future data')

# optimization
parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
parser.add_argument('--itr', type=int, default=2, help='experiments times')
parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
parser.add_argument('--patience', type=int, default=7, help='early stopping patience')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
parser.add_argument('--des', type=str, default='test', help='exp description')
parser.add_argument('--loss', type=str, default='mse', help='loss function')
parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

# GPU
parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
parser.add_argument('--gpu', type=int, default=7, help='gpu')
parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')
parser.add_argument('--test_flop', action='store_true', default=False, help='See utils/tools for usage')
# GNN
parser.add_argument('--blocks', type=int, default=1, help='gpu')
parser.add_argument('--tvechidden', type=int, default=50, help='gpu')
parser.add_argument('--nvechidden', type=int, default=20, help='gpu')
parser.add_argument('--use_tgcn', type=int, default=1, help='gpu')
parser.add_argument('--use_ngcn', type=int, default=1, help='gpu')
parser.add_argument('--anti_ood', type=int, default=1, help='gpu')
parser.add_argument('--scale_number', type=int, default=4, help='gpu')
parser.add_argument('--hidden', type=int, default=16, help='gpu')
parser.add_argument('--tk', type=int, default=10, help='gpu')

args,unknown = parser.parse_known_args()
# import sys
# sys.path.append('/home/hqh/Transformer/LTSF-Linear-main/FEDformer/')
from data_provider.data_factory import data_provider
def _get_data(args,flag):
    #sys.path.append('/home/hqh/Transformer/LTSF-Linear-main/')
    data_set, data_loader = data_provider(args, flag)
    return data_set, data_loader

def get_args():
    return args
def build_model(args,model_name):
    model_dict = {
        'Autoformer': Autoformer,
        'Transformer': Transformer,
        'Informer': Informer,
        'DLinear': DLinear,
        'NLinear': NLinear,
        'Linear': Linear,
        'hqhGNN': hqhGNN,
        'GNNLinear':GNNLinear,
        'CrossTwo':CrossTwo,
        'CrossFFTGNN':CrossFFTGNN
    }
    model = model_dict[model_name].Model(args).float()
    return model

