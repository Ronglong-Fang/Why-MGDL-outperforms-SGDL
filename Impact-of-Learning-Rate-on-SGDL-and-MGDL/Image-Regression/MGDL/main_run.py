from mgdl import MGDLmodel, data_setup
import jax.numpy as np
import pickle
from argparse import Namespace

learning_rate = 1e-2                      #set a learning rate

opt = Namespace()
opt.image = 'resolutionchart'             #set the name of figure: resolutionchart, cameraman, barbara, male



data, opt = data_setup(opt)              #generate data

opt.epoch = 100000                       #set the number of epoch.


opt.activation = 'relu'                  #activation function for MGDL
opt.loss_record = 10
opt.loss_smooth = 20
opt.rel_error = 1e-6
opt.learning_rate = learning_rate




#---------------structure for MGDL---------------
opt.grade = 4
opt.num_channel = {}
opt.num_channel['grade1'] = [2, 128, 128, 1]
opt.num_channel['grade2'] = [128, 128, 128, 1]
opt.num_channel['grade3'] = [128, 128, 128, 1]
opt.num_channel['grade4'] = [128, 128, 128, 1]
#-------------------------------------------------


#------------train MGDL model---------------------
MGDLmodel(opt, data)
#-------------------------------------------------
    
    
    




