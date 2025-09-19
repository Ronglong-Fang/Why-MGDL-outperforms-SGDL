from mgdl import MGDLmodel, data_setup
import jax.numpy as np
import pickle
from argparse import Namespace

grade = 4                                  #the number of grade for MGDL

#-----------model parameter-----
learning_rate = 5e-5
beta = 1e-3
lambd = 1e-4
#--------------------------------


noise_level = 10/255                       #noise level: 10/255, 30/255
image = 'butterfly'                        #image: butterfly, barbara


opt = Namespace()
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
opt.interval = 5000
opt.eig = True

#--------------strcuture for MGDL------------
opt.grade = grade
opt.num_channel = {}
opt.num_channel['grade1'] = [2, 48, 1]
opt.num_channel['grade2'] = [48, 48, 1]
opt.num_channel['grade3'] = [48, 48, 1]
opt.num_channel['grade4'] = [48, 48, 1]
#--------------------------------------------

#-------------train MGDL---------------------
MGDLmodel(opt, data)
#--------------------------------------------







