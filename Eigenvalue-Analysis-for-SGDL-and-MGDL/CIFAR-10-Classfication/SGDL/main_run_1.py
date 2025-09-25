from sgdl import train_model, data_setup
import jax.numpy as np
import pickle
from argparse import Namespace

Learning_rate = [4e-3]
for lr in Learning_rate:
    learning_rate = lr                     #set a learning rate
    
    opt = Namespace()
    
    
    
    data, opt = data_setup(opt)               #generate data
    
    opt.epoch = 400000                        #set the number of epochs.
    
    opt.activation = 'relu'                   #activation function for SGDL
    opt.loss_record = 100
    opt.loss_smooth = 20
    opt.rel_error = 1e-4
    opt.learning_rate = learning_rate
    opt.interval = 100
    
    opt.eig = True
    
    #---------------structure for SGDL---------------
    opt.num_channel = [3072, 128, 128, 128, 128, 128, 128, 128, 128, 10]
    #-------------------------------------------------
    
    #------------train SGDL model---------------------
    train_data = [data["train_X"], data["train_Y"]]
    val_data = [data["val_X"], data["val_Y"]]
    train_model(opt, train_data, val_data)
    #-------------------------------------------------
    
    
        
        
        
    
    
    

