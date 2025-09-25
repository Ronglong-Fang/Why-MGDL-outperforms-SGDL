from transformer_utils import MGDLmodel
from argparse import Namespace

opt = Namespace()

# -------------------------------
# data
# -------------------------------
opt.T = 64
opt.n_total = 2000
opt.n_train = int(0.8*opt.n_total)
opt.noise_std = 0

# -------------------------------
# model paramater
# -------------------------------
opt.d_model = 64
opt.n_layers = 1
opt.n_heads = 1
opt.grade = 3
opt.d_ff = 128
opt.dropout = 0.0
opt.activation = 'relu'

# -------------------------------
# optimization paramater
# -------------------------------
opt.batch_size = 128
opt.epochs = 300
opt.lr = {
    'grade1': 1e-4,'grade2': 1e-4,'grade3': 1e-4
}
opt.seed = 0
opt.loss_record = 1   
opt.interval = 1     

# -------------------------------
# train model
# -------------------------------
MGDLmodel(opt)

