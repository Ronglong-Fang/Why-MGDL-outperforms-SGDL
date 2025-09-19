from sgdl import train_model, data_setup
import jax.numpy as np
import pickle
from argparse import Namespace

learning_rate = 1e-2                      #set a learning rate

opt = Namespace()
opt.image = 'resolutionchart'             #set the name of image: resolutionchart, cameraman, barbara, butterfly



data, opt = data_setup(opt)               #generate data

opt.epoch = 500000                        #set the number of epochs.

opt.activation = 'relu'                  #activation function for SGDL
opt.loss_record = 100
opt.loss_smooth = 20
opt.rel_error = 1e-4
opt.learning_rate = learning_rate
opt.interval = 10000

#---------------structure for SGDL---------------
opt.num_channel = [2, 48, 48, 48, 48, 1]
#-------------------------------------------------

#------------train SGDL model---------------------
train_data = [data["train_X"], data["train_Y"]]
val_data = [data["val_X"], data["val_Y"]]
test_data = [data["test_X"], data["test_Y"]]
train_model(opt, train_data, val_data, test_data)
#-------------------------------------------------


    
    
    




