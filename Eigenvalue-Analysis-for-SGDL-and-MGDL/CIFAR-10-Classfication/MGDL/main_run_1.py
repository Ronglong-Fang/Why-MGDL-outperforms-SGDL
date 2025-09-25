from mgdl import MGDLmodel, data_setup
import jax.numpy as np
import pickle
from argparse import Namespace


Learning_rate = [1e-3]
for lr in Learning_rate:
    learning_rate = lr  
    opt = Namespace()
    data, opt = data_setup(opt)                           #generate data
    
    opt.epoch = {
        'grade1': 300000,
        'grade2': 1000000,
        'grade3': 2000000,
        'grade4': 2000000
    }
    
    opt.activation = 'relu'                               #activation function for MGDL
    opt.loss_record = 1000
    opt.loss_smooth = 20
    opt.rel_error = 1e-5
    opt.learning_rate = learning_rate
    opt.interval =  {
        'grade1': 3000,
        'grade2': 5000,
        'grade3': 5000,
        'grade4': 5000 
    }
    opt.eig = True
    
    #---------------structure for MGDL---------------
    opt.grade = 4
    opt.num_channel = {}
    opt.num_channel['grade1'] = [3072, 128, 128, 10]
    opt.num_channel['grade2'] = [128, 128, 128, 10]
    opt.num_channel['grade3'] = [128, 128, 128, 10]
    opt.num_channel['grade4'] = [128, 128, 128, 10]
    #-------------------------------------------------
    
    
    #------------train MGDL model---------------------
    MGDLmodel(opt, data)
    #-------------------------------------------------
        
    
        
        
        
    
    
    

