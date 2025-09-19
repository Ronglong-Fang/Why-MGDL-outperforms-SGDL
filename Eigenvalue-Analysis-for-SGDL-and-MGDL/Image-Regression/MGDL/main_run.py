from mgdl import MGDLmodel, data_setup
import jax.numpy as np
import pickle
from argparse import Namespace

learning_rate = 1e-1                                  #set a learning rate
opt = Namespace()
opt.image = 'resolutionchart'                         #set the name of image: resolutionchart, cameraman, barbara, butterfly


data, opt = data_setup(opt)                           #generate data

opt.epoch = 500000                                    #set the number of epochs.

opt.activation = 'relu'                               #activation function for MGDL
opt.loss_record = 100
opt.loss_smooth = 20
opt.rel_error = 1e-5
opt.learning_rate = learning_rate
opt.interval = 10000


#---------------structure for MGDL---------------
opt.grade = 4
opt.num_channel = {}
opt.num_channel['grade1'] = [2, 48, 1]
opt.num_channel['grade2'] = [48, 48, 1]
opt.num_channel['grade3'] = [48, 48, 1]
opt.num_channel['grade4'] = [48, 48, 1]
#-------------------------------------------------


#------------train MGDL model---------------------
MGDLmodel(opt, data)
#-------------------------------------------------
    

    
    
    




