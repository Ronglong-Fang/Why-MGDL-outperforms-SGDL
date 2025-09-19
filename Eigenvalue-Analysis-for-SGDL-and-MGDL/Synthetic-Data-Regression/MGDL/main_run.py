from mgdl import MGDLmodel
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
opt.eig = True

#---------------structure for MGDL---------------
opt.grade = 4
opt.num_channel = {}
opt.num_channel['grade1'] = [1, 32, 1]
opt.num_channel['grade2'] = [32, 32, 1]
opt.num_channel['grade3'] = [32, 32, 1]
opt.num_channel['grade4'] = [32, 32, 1]
#-------------------------------------------------


#------------train MGDL model---------------------
MGDLmodel(opt, data)
#-------------------------------------------------



    
    
    




