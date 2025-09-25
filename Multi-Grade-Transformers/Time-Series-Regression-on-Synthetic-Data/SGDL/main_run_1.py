from transformer_utils import train_model
from argparse import Namespace


opt = Namespace()

# -------------------------------
# data
# -------------------------------
opt.T = 64
opt.n_total = 2000
opt.noise_std = 0

# -------------------------------
# model paramater
# -------------------------------
opt.d_model = 64
opt.n_layers = 3
opt.n_heads = 1
opt.d_ff = 128
opt.dropout = 0.0
opt.activation = 'relu'

# -------------------------------
# optimization paraamter
# -------------------------------
opt.batch_size = 128
opt.epochs = 300
opt.lr = 1e-3
opt.seed = 0
opt.loss_record = 1  
opt.interval = 1     



# -------------------------------
# train model
# -------------------------------
train_model(opt)

