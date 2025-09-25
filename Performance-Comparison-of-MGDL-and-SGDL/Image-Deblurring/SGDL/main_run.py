import matplotlib.pyplot as plt
from mgdlmodel import MGDLmodel, data_setup
from argparse import Namespace
from jax import random
import pickle

rand_key = random.PRNGKey(0)

grade = 1                                  #SGDL is an MGDL with grade set to be 1
layer = 12                                 #the number of layers
width = 128                                #the number of neural in each layer

epoch = 20000                              #the number of train epoch

#-----------model parameter-----
lr_params = 1e-3
beta = 1e-1
lambd = 1e-2
#--------------------------------

noise_level = 3/255                       #noise level: 10/255, 20/255, ..., 60/255
blurring_level = 3                        #blurring level: 3, 5, 7
image = 'butterfly'                       #image: butterfly, male, chest




opt = Namespace()
opt.noise_level = noise_level
opt.image = image
opt.sigma = blurring_level
opt.size = 6*opt.sigma+1
data = data_setup(opt)

opt.epoch = epoch
opt.num_channel = width
opt.num_layer = layer
opt.activation = 'relu'


opt.lr_params =  lr_params
opt.beta = beta
opt.lambd = lambd

opt.grade = grade
opt.alpha = 0.99

opt.interval = 100



    
MGDLmodel(opt, data)
