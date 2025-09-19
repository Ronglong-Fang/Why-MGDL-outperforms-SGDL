import numpy
import jax
import jax.numpy as np
from jax import jit, grad, random
from jax.example_libraries import stax, optimizers
import pickle
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import os, imageio
from jax.scipy.signal import convolve
from Hessian import compute_Hessian_layer4_scale4
from SpectralBiasData import generate_data

def create_network(opt):
    
    def model_fn(params, inputs, **kwargs):
        z = {}
        a = {}
        for i in range(1, opt.layer):
            z[i] = {}
            a[i] = {}
            
        #loop scale
        for scale in range(1, opt.scale+1):
            #loop layer
            for i in range(1, opt.layer):
                w, b = params['scale'+str(scale)][i-1]
                if i==1:
                    z[i]['scale'+str(scale)] = np.dot(w.T, opt.coeff['scale'+str(scale)]*inputs) + b
                    a[i]['scale'+str(scale)] = np.maximum(z[i]['scale'+str(scale)], 0)  
                else:
                    z[i]['scale'+str(scale)] = np.dot(w.T, a[i-1]['scale'+str(scale)]) + b
                    a[i]['scale'+str(scale)] = np.maximum(z[i]['scale'+str(scale)], 0)  
                    
            w, b = params['scale'+str(scale)][i]
            if scale == 1:
                output = np.dot(w.T, a[i]['scale'+str(scale)]) + b
            else:
                output += np.dot(w.T, a[i]['scale'+str(scale)]) + b
        
        return output, z, a

    def he_init(key, shape):
        fan_in = shape[0]  # Number of input neurons
        std = np.sqrt(2.0 / fan_in)
        return jax.random.normal(key, shape) * std
    
    def init_params():
        key = jax.random.PRNGKey(42)
        params = {}
        #loop scale
        for scale in range(1, opt.scale+1):
            params['scale'+str(scale)] = []
            #loop layer
            for i in range(1, opt.layer+1):
                key, subkey = jax.random.split(key)
                w = he_init(subkey, (opt.num_channel[i-1], opt.num_channel[i]))
                key, subkey = jax.random.split(key)
                b = np.zeros((opt.num_channel[i], 1))
                params['scale'+str(scale)].append((w, b))
    
        return params


    return model_fn, init_params



# Train model with given hyperparameters and data
def train_model(opt, train_data, val_data, test_data):

    key = random.PRNGKey(0)
    
    s_time = time.time() 
    model_fn, init_params = create_network(opt)
    model_pred = jit(lambda params, x : model_fn(params, x)[0])   
    model_loss = jit(lambda params, x, y: .5 * np.mean((model_pred(params, x) - y) ** 2))
    model_grad_loss = jit(lambda params, x, y: grad(model_loss)(params, x, y))
    
    opt_init, opt_update, get_params = optimizers.sgd(opt.learning_rate)

    opt_update = jit(opt_update)


    params = init_params()

    opt_state = opt_init(params)
    train_loss = []
    val_loss = []
    val_loss_smooth = []

    
    xs = []

    eig_Hessian = []

    for i in tqdm(range(opt.epoch)):

        if i%opt.interval==0:
            params = get_params(opt_state)
            output, z, a = model_fn(params, train_data[0])   
            Hessian = compute_Hessian_layer4_scale4(opt, params, z, a, output, train_data[0], train_data[1])
            eigenvalues, _ = np.linalg.eigh(np.eye(len(Hessian)) - opt.learning_rate * Hessian) 
            eig_Hessian.append(eigenvalues)

            print(f"shape Hessian: {np.shape(Hessian)}")
            print(f"epoch is {i}, eigenvalue of hessian is {eigenvalues}")
                    

        opt_state = opt_update(i, model_grad_loss(get_params(opt_state), train_data[0], train_data[1]), opt_state)

        if i % opt.loss_record == 0:
            train_loss.append( model_loss(get_params(opt_state), *train_data) )
            val_loss.append( model_loss(get_params(opt_state), *val_data))
            xs.append(i)
            

        if (i+1) % (opt.loss_record * opt.loss_smooth) == 0:
            val_loss_smooth.append(np.mean(np.array(val_loss[-opt.loss_smooth:])))

            if len(val_loss_smooth)>1:
                rel_error = np.abs((val_loss_smooth[-2]-val_loss_smooth[-1])/val_loss_smooth[-2])
                if rel_error < opt.rel_error:
                    break

                    
    e_time = time.time()
    
    test_loss =  model_loss(get_params(opt_state), *test_data)  
    train_pred = model_pred(get_params(opt_state), train_data[0])
    val_pred = model_pred(get_params(opt_state), val_data[0])
    test_pred = model_pred(get_params(opt_state), test_data[0])
        

    history =  {
        'params': get_params(opt_state), 
        'xs': xs,
        'train_pred': train_pred,
        'val_pred': val_pred,
        'test_pred': test_pred,
        'train_loss': train_loss,
        'val_loss': val_loss,
        'test_loss': test_loss,
        'train_time': e_time - s_time,
        'train_epoch': i,
        'eig_Hessian': eig_Hessian
    }


    picklename = 'results/SGDLacti%s_epoch%d_learningrate%.4e_trainloss%.4e_valloss%.4e_testloss%.4e.pickle' %(
        opt.activation, opt.epoch, opt.learning_rate, history['train_loss'][-1], history['val_loss'][-1], history['test_loss']
        )
    
    with open(picklename, 'wb') as f:
        pickle.dump([history, opt], f)  


    return 






def analysis(filepath):


    with open(filepath, 'rb') as f:
        [history, opt] = pickle.load(f)


    data, _ = generate_data(opt.Amptype, opt.J)  
    print(opt)


    nshow = 10

    # plt.figure(figsize=(10, 6))
    plt.plot(history['xs'], history['train_loss'], label="Training loss")
    plt.plot(history['xs'], history['train_loss'], '--', label="Validation loss")
    plt.xlabel('Epochs', fontsize=20)
    plt.ylabel('Loss', fontsize=20)
    plt.title(f"Multi-Scale", fontsize=20)
    plt.yscale('log')
    plt.ylim([5e-5, 1e0])
    plt.legend(fontsize=20)             
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    fig_filename = f'Fig/MultiScale_FrequencyFunction1_Loss.png'
    plt.savefig(fig_filename, format='png', bbox_inches='tight')
    plt.show()

    print(f"train time: {history['train_time']}, epoch: {history['train_epoch']}, train loss: {history['train_loss'][-1]} val loss: {history['val_loss'][-1]}, test loss: {history['test_loss']}")

    # plt.figure(figsize=(10, 6))
    plt.plot(data["train_X"].T, data["train_Y"].T)
    plt.plot(data["train_X"].T, history["train_pred"].T)
    plt.xlabel("$x$", fontsize=20)
    plt.ylabel("$y$", fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.title(f"Multi-Scale", fontsize=20)
    fig_filename = f'Fig/MultiScale_FrequencyFunction1_Predict.png'
    plt.savefig(fig_filename, format='png', bbox_inches='tight')
    plt.show()       

    Eig_dict_MIN = {}
    for j in range(nshow):
        Eig_dict_MIN['Index'+str(j)] = []

    for j in range(nshow):
        for i in range(0, len(history["eig_Hessian"])):
            Eig_dict_MIN['Index'+str(j)].append(history["eig_Hessian"][i][j])
    
    Eig_dict_MAX = {}
    for j in range(nshow):
        Eig_dict_MAX['Index'+str(j)] = []

    for j in range(nshow):
        for i in range(0, len(history["eig_Hessian"])):
            Eig_dict_MAX['Index'+str(j)].append(history["eig_Hessian"][i][len(history["eig_Hessian"][i])-nshow+j])

    # plt.figure(figsize=(10, 6))
    for j in range(nshow):
        plt.plot(range(0, history['xs'][-1], opt.interval), np.array(Eig_dict_MIN['Index'+str(j)]).T, label=f"Index {j}")
        plt.plot(range(0, history['xs'][-1], opt.interval), 1 * numpy.ones_like(numpy.array(range(0, history['xs'][-1], opt.interval))), 'r--')
        plt.plot(range(0, history['xs'][-1], opt.interval), -1 * numpy.ones_like(numpy.array(range(0, history['xs'][-1], opt.interval))), 'r--')
        

    for j in range(nshow):
        
        plt.plot(range(0, history['xs'][-1], opt.interval), np.array(Eig_dict_MAX['Index'+str(j)]).T, linestyle='--', label=f"Index {nshow+j}")


    plt.legend(fontsize=9, loc='upper left', bbox_to_anchor=(1, 1))
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.ylim([-1.8, 1.2])
    plt.xlabel('Epoch', fontsize=20)
    plt.ylabel('Eigenvalues', fontsize=20)
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    plt.title('Multi-Scale: Eigenvalues of $\mathbf{I} - \eta\mathbf{H}_{\mathcal{L}}(\mathbf{W}^k)$', fontsize=17)
    fig_filename = f'Fig/MultiScale_FrequencyFunction1_Eig.png'
    plt.savefig(fig_filename, format='png', bbox_inches='tight')
    plt.show()

    # plt.figure(figsize=(10, 6))
    # for j in range(nshow):
        
    #     plt.plot(range(0, history['xs'][-1], opt.interval), np.array(Eig_dict_MAX['Index'+str(j)]).T, 
    #              label=f"Index {j}")
    #     plt.plot(range(0, history['xs'][-1], opt.interval), 1 * numpy.ones_like(numpy.array(range(0, history['xs'][-1], opt.interval))), 'r--')
    #     plt.xlabel('Epoch', fontsize=20)
    #     plt.ylabel('Eigenvalues', fontsize=20)
    # plt.ylim([0.99, 1.09])
    # plt.legend(fontsize=20, loc='upper left', bbox_to_anchor=(1, 1))
    # plt.xticks(fontsize=20)
    # plt.yticks(fontsize=20)
    # plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    # plt.title('Single-Grade: Ten largest eigenvalues of $\mathbf{I} - \eta\mathbf{H}_f(\mathbf{W}^k)$', fontsize=20)
    # fig_filename = f"Fig/MultiScale_FrequencyFunction1_TenLargest.png"
    # plt.savefig(fig_filename, format='png',  bbox_inches='tight')
    # plt.show()