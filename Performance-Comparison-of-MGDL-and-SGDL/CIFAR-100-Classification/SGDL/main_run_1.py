from sgdl import train_model, data_setup
import jax.numpy as np
import pickle
from argparse import Namespace

Learning_rate = [1e-4]
for lr in Learning_rate:
    learning_rate = lr                     #set a learning rate
    
    opt = Namespace()
    
    data, opt = data_setup(opt)               #generate data
    
    opt.epoch = 10000                          #set the number of epochs.
    
    opt.activation = 'relu'                   #activation function for SGDL
    opt.loss_record = 10
    opt.learning_rate = learning_rate
    opt.interval = 10
    opt.batch_size = 128
    
    opt.eig = False
    
    
    #------------train SGDL model---------------------
    train_data = [data["train_X"], data["train_Y"]]
    val_data = [data["val_X"], data["val_Y"]]
    train_model(opt, train_data, val_data)
    #-------------------------------------------------
    
    
        
        
        
    
    
    

