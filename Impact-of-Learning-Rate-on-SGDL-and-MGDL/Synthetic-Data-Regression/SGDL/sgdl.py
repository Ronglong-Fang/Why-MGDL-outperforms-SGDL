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
from Hessian import compute_Hessian_four

def create_network(opt):
    
    def model_fn(params, inputs, **kwargs):
        w1, b1 = params[0]
        z1 = np.dot(w1.T, inputs) + b1
        a1 = np.maximum(z1, 0)  

        w2, b2 = params[1]
        z2 = np.dot(w2.T, a1) + b2
        a2 = np.maximum(z2, 0) 

        w3, b3 = params[2]
        z3 = np.dot(w3.T, a2) + b3
        a3 = np.maximum(z3, 0)  
        
        w4, b4 = params[3]
        z4 = np.dot(w4.T, a3) + b4
        a4 = np.maximum(z4, 0)  

        w5, b5 = params[4]
        output = np.dot(w5.T, a4) + b5
        
        return output, z1, z2, z3, z4, a1, a2, a3, a4



    def he_init(key, shape):
        fan_in = shape[0]  # Number of input neurons
        std = np.sqrt(2.0 / fan_in)
        return jax.random.normal(key, shape) * std
    
    def init_params():
        key = jax.random.PRNGKey(42)
        
        key, subkey = jax.random.split(key)
        w1 = he_init(subkey, (opt.num_channel[0], opt.num_channel[1]))
        key, subkey = jax.random.split(key)
        b1 = np.zeros((opt.num_channel[1], 1))
    
        key, subkey = jax.random.split(key)
        w2 = he_init(subkey, (opt.num_channel[1], opt.num_channel[2]))
        key, subkey = jax.random.split(key)
        b2 = np.zeros((opt.num_channel[2], 1))
    
        key, subkey = jax.random.split(key)
        w3 = he_init(subkey, (opt.num_channel[2], opt.num_channel[3]))
        key, subkey = jax.random.split(key)
        b3 = np.zeros((opt.num_channel[3], 1))

        key, subkey = jax.random.split(key)
        w4 = he_init(subkey, (opt.num_channel[3], opt.num_channel[4]))
        key, subkey = jax.random.split(key)
        b4 = np.zeros((opt.num_channel[4], 1))

        key, subkey = jax.random.split(key)
        w5 = he_init(subkey, (opt.num_channel[4], opt.num_channel[5]))
        key, subkey = jax.random.split(key)
        b5 = np.zeros((opt.num_channel[5], 1))

        params = [(w1, b1), (w2, b2), (w3, b3), (w4, b4), (w5, b5)]
    
       
    
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

        
        if opt.eig:
            if i%opt.interval==0:
                params = get_params(opt_state)
                output, z1, z2, z3, z4, a1, a2, a3, a4 = model_fn(params, train_data[0])   
                Hessian = compute_Hessian_four(opt, params, z1, z2, z3, z4, a1, a2, a3, a4, output, train_data[0], train_data[1])
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
        

    if opt.eig:
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
    else:
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
            'train_epoch': i
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

    
    if opt.setting == 1:
        from SpectralBiasDataSetting1 import generate_data
    else opt.setting == 2:
        from SpectralBiasDataSetting2 import generate_data

    data, _ = generate_data(opt.Amptype, opt.J)  
    print(opt)


    nshow = 10

    # plt.figure(figsize=(10, 6))
    plt.plot(history['xs'], history['train_loss'], label="Training loss")
    plt.plot(history['xs'], history['train_loss'], '--', label="Validation loss")
    plt.xlabel('Epochs', fontsize=20)
    plt.ylabel('Loss', fontsize=20)
    plt.title(f"Single-Grade", fontsize=20)
    plt.yscale('log')
    plt.ylim([7e-4, 2e0])
    plt.legend(fontsize=20)             
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    fig_filename = f'Fig/SingleGrade_FrequencyFunction2_Loss.png'
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
    plt.title(f"Single-Grade", fontsize=20)
    fig_filename = f'Fig/SingleGrade_FrequencyFunction2_Predict.png'
    plt.savefig(fig_filename, format='png', bbox_inches='tight')
    plt.show()       

    if opt.eig:
        
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
        plt.ylim([-2.7, 1.2])
        plt.xlabel('Epochs', fontsize=20)
        plt.ylabel('Eigenvalues', fontsize=20)
        plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
        plt.title('Single-Grade: Eigenvalues of $\mathbf{I} - \eta\mathbf{H}_{\mathcal{L}}(\mathbf{W}^k)$', fontsize=17)
        fig_filename = f'Fig/SingleGrade_FrequencyFunction2_Eig.png'
        plt.savefig(fig_filename, format='png', bbox_inches='tight')
        plt.show()
