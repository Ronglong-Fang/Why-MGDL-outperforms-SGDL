from sgdltwoLayerModifiedwithBias import train_model
from SpectralBiasData import generate_data
import jax.numpy as np
import pickle
from argparse import Namespace

setting = 1                   #the number of setting: 1, 2
learning_rate = 1e-2          #set a learning rate
Amptype = 'constant'
J = 10


if setting == 1:
    from SpectralBiasDataSetting1 import generate_data
else setting == 2:
    from SpectralBiasDataSetting2 import generate_data

data, opt = generate_data(Amptype, J)    # generate data 

opt.setting = setting
opt.epoch = 1000000

opt.activation = 'relu'
opt.loss_record = 100
opt.loss_smooth = 20
opt.rel_error = 1e-6
opt.learning_rate = learning_rate
opt.interval = 10000
opt.eig = False

#---------------structure for MGDL---------------
opt.num_channel = [1, 32, 32, 32, 32, 1]
#-------------------------------------------------


#------------train MGDL model---------------------
train_data = [data["train_X"], data["train_Y"]]
val_data = [data["val_X"], data["val_Y"]]
test_data = [data["test_X"], data["test_Y"]]
train_model(opt, train_data, val_data, test_data)
#-------------------------------------------------


    
    
    




