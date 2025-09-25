from mgdl import MGDLmodel, data_setup
import jax.numpy as np
import pickle
from argparse import Namespace


Learning_rate = [1e-4]
for lr in Learning_rate:
    learning_rate = lr  
    opt = Namespace()
    data, opt = data_setup(opt)                           #generate data
    
    opt.epoch = {
        'grade1': 10000,
        'grade2': 10000,
        'grade3': 10000,
        'grade4': 10000
    }

    opt.activation = 'relu'                               #activation function for MGDL
    opt.loss_record = 10
    opt.learning_rate = learning_rate
    opt.interval =  {
        'grade1': 10,
        'grade2': 10,
        'grade3': 10,
        'grade4': 10
    }
    opt.batch_size = 128
    opt.eig = False
    
    #---------------structure for MGDL---------------
    opt.grade = 4
    #------------------------------------------------
    
    
    #------------train MGDL model---------------------
    MGDLmodel(opt, data)
    #-------------------------------------------------
        
    
        
        
        
    
    
    

