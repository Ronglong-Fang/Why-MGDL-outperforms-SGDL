import numpy as np
import matplotlib.pyplot as plt
from argparse import Namespace
import jax
import jax.numpy as np
from jax import random


def Spectral_bias_constantORincrease_amplituide(opt):
    """
    generate sample train data and test data
    this example is from the paper : On the Spectral Bias of Neural Networks

    Returns
    -------
    None.

    """
    # training
    train_X = np.reshape(np.linspace(0, 1, opt.ntrain), (1, opt.ntrain))
    train_Y = np.zeros((1, opt.ntrain))    
    # validation 
    key = random.PRNGKey(1)  # JAX's way of handling random state
    val_X = random.uniform(key, shape=(1, opt.nval))
    val_Y = np.zeros((1, opt.nval))   
    # testing
    test_X = np.reshape(np.linspace(0, 1, opt.ntest), (1, opt.ntest))
    test_Y = np.zeros((1, opt.ntest))    


    for i in range(len(opt.kappa)):
        train_Y += opt.alpha[i]*np.sin(2*np.pi*opt.kappa[i]*train_X + opt.phi[i])
        val_Y += opt.alpha[i]*np.sin(2*np.pi*opt.kappa[i]*val_X + opt.phi[i])
        test_Y += opt.alpha[i]*np.sin(2*np.pi*opt.kappa[i]*test_X + opt.phi[i])
    
    data = {}
    data['opt'] = opt    
    data['train_X'] = train_X
    data['train_Y'] = train_Y
    data['val_X'] = val_X
    data['val_Y'] = val_Y
    data['test_X'] = test_X
    data['test_Y'] = test_Y

    plt.plot(train_X.T, train_Y.T)
    plt.show()

    return data





def generate_data(Amptype, J):

    kk=5

    opt = Namespace()
    opt.Amptype = Amptype
    opt.J = J
    opt.ntrain = 2**J
    opt.nval = 2**J
    opt.ntest = 10000
    # opt.kappa = np.linspace(20, 400, 20)
    # opt.kappa = np.linspace(5, 100, 20)
    opt.kappa = np.linspace(1, 30, kk)
    key = jax.random.PRNGKey(0)
    opt.phi = jax.random.uniform(key, shape=(kk,), minval=0, maxval=2 * np.pi)
    opt.alpha = np.linspace(1, 1, kk)

    if Amptype == "constant":
        opt.alpha = np.linspace(1, 1, 20)
        data = Spectral_bias_constantORincrease_amplituide(opt)       
    elif Amptype == "decrease":
        opt.alpha = np.linspace(1, 0.05, 20)
        data = Spectral_bias_constantORincrease_amplituide(opt)
    elif Amptype == "increase":
        opt.alpha = np.linspace(0.05, 1, 20)
        data = Spectral_bias_constantORincrease_amplituide(opt)
    elif Amptype == "vary":
        data = Spectral_bias_vary_amplituide(opt)

    return data, opt
    



def fft(yt):
    
    n = len(np.squeeze(yt)) # length of the signal
    frq = np.arange(n)
    frq = frq[range(n//2)] # one side frequency range
    # -------------
    FFTYT = np.squeeze(np.fft.fft(yt)/n) # fft computing and normalization

    FFTYT = FFTYT[range(n//2)]
    fftyt = abs(FFTYT)


    # plt.plot(frq, fftyt)
    # plt.xlim(0, 50)
    # plt.xlabel("Frequency [Hz]")
    # plt.ylabel("Amplitude")
    # plt.show()

    return frq, fftyt




    



