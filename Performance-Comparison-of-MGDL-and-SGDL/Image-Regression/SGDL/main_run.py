from sgdl import train_model, data_setup
import jax.numpy as np
import pickle
from argparse import Namespace


learning_rate = 1e-3                   #set a learning rate

opt = Namespace()
opt.image = 'cameraman'                #set the name of figure: cameraman, barbara, butterfly, male, chest, walnut



data, opt = data_setup(opt)            #generate data

opt.epoch = 10000                      #set the number of epoch.

opt.activation = 'relu'                #activation function for MGDL
opt.loss_record = 10
opt.loss_smooth = 20
opt.rel_error = 1e-4
opt.learning_rate = learning_rate

#---------------structure for SGDL---------------
opt.num_channel = [2, 128, 128, 128, 128, 128, 128, 128, 128, 1]



#------------train SGDL model---------------------
train_data = [data["train_X"], data["train_Y"]]
val_data = [data["val_X"], data["val_Y"]]
test_data = [data["test_X"], data["test_Y"]]
train_model(opt, train_data, val_data, test_data)


    
    
    




