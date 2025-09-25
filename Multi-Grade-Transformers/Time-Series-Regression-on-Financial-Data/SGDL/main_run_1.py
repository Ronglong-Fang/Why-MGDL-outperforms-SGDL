from transformer_utils import train_model
from argparse import Namespace


opt = Namespace()

# -------------------------------
# data
# -------------------------------
opt.T = 20

# -------------------------------
# 模型参数
# -------------------------------
opt.d_model = 64
opt.n_layers = 6
opt.n_heads = 1
opt.d_ff = 128
opt.dropout = 0.0
opt.activation = 'relu'

# -------------------------------
# 训练参数
# -------------------------------
opt.batch_size = 128
opt.epochs = 30
opt.lr = 1e-3
opt.seed = 0
opt.loss_record = 1   



# -------------------------------
# 训练模型
# -------------------------------
train_model(opt)

