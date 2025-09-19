from sgdltwoLayerModifiedwithBias import train_model, data_setup
import jax.numpy as np
import pickle
from argparse import Namespace

#-----------model parameter-----
learning_rate = 1e-5
beta = 1e-2
lambd = 1e-1
#--------------------------------

noise_level = 10/255                       #noise level: 10/255, 30/255
image = 'butterfly'                        #image: butterfly, barbara

opt = Namespace()
noise_level = 10/255
opt.noise_level = noise_level
opt.image = image

data, opt = data_setup(opt)                #generate data

opt.epoch = 100000                         #the number of training epoch


opt.activation = 'relu'                    #activation function
opt.loss_record = 100
opt.learning_rate = learning_rate
opt.beta = beta
opt.lambd = lambd
opt.alpha = 0.99
opt.interval = 2000
opt.eig = True

#--------------strcuture for SGDL------------
opt.num_channel = [2, 48, 48, 48, 48, 1]
#--------------------------------------------

#-------------train SGDL---------------------
train_model(opt, data)
#--------------------------------------------
        
            
    
    




