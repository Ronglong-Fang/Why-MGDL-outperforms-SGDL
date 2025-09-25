from transformer_utils import MGDLmodel
from argparse import Namespace

opt = Namespace()

# -------------------------------
# 数据参数
# -------------------------------
opt.T = 20

# -------------------------------
# 模型参数
# -------------------------------
opt.d_model = 64
opt.n_layers = 1
opt.n_heads = 1
opt.grade = 6
opt.d_ff = 128
opt.dropout = 0.0
opt.activation = 'relu'

# -------------------------------
# 训练参数
# -------------------------------
opt.batch_size = 128
opt.epochs = 30
opt.lr = {
    'grade1': 1e-3,'grade2': 1e-3,
    'grade3': 1e-3,'grade4': 1e-3,
    'grade5': 1e-3,'grade6': 1e-3
}
opt.seed = 0
opt.loss_record = 1   # 每多少步记录 loss



# -------------------------------
# 训练模型
# -------------------------------
MGDLmodel(opt)

