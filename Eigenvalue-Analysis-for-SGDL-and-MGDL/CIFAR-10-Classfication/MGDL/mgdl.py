import numpy
import jax
import jax.numpy as jnp
from jax import jit, grad, random
from jax.example_libraries import stax, optimizers
import pickle
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import os, imageio
from jax.scipy.signal import convolve
from tensorflow.keras.datasets import cifar10
import optax
from jax import flatten_util, jvp
from jax.nn import one_hot



#set up data
def data_setup(opt):

    # Load data
    (train_X, train_Y), (test_X, test_Y) = cifar10.load_data()

    num_classes = 10  # or 2 if you're doing binary classification like 7 vs 9
    
    train_Y = one_hot(train_Y, num_classes)
    test_Y = one_hot(test_Y, num_classes)

    # Use your fixed random key for reproducibility
    rand_key = random.PRNGKey(0)
    
    # Limit to 5000 training samples (randomly selected)
    num_samples = 10000
    indices = random.choice(rand_key, len(train_X), shape=(num_samples,), replace=False)
    
    train_X = train_X[indices]
    train_Y = train_Y[indices]

    train_X = train_X.astype('float32') / 255.0
    test_X = test_X.astype('float32') / 255.0

    train_X = train_X.reshape(train_X.shape[0], -1)
    test_X = test_X.reshape(test_X.shape[0], -1)


    train_Y = train_Y.reshape(train_Y.shape[0], -1)
    test_Y = test_Y.reshape(test_X.shape[0], -1)

    print(f"train_X shape: {train_X.shape}, dtype: {train_X.dtype}")
    print(f"test_X shape: {test_X.shape}, dtype: {test_X.dtype}")

    print(f"train_Y shape: {train_Y.shape}, dtype: {train_Y.dtype}")
    print(f"test_Y shape: {test_Y.shape}, dtype: {test_Y.dtype}")    


    data = {}
    data['train_X'] = train_X.T
    data['train_Y'] = train_Y.T
    data['val_X'] = test_X.T
    data['val_Y'] = test_Y.T

    opt.ntrain = train_X.T.shape[1]

    return data, opt



def create_network(opt, grade):
    
    def model_fn(params, inputs, **kwargs):
        x = inputs
        num_layers = len(params)
        for i in range(num_layers - 1):
            w, b = params[i]
            x = jnp.dot(w.T, x) + b  
            x = jnp.maximum(x, 0)    
        w_last, b_last = params[-1]
        output = jnp.dot(w_last.T, x) + b_last
        return output, x

    def he_init(key, shape):
        fan_in = shape[0]
        std = jnp.sqrt(2.0 / fan_in)
        return jax.random.normal(key, shape) * std

    def init_params():
        key = jax.random.PRNGKey(42)
        params = []
        for i in range(len(opt.num_channel["grade"+str(grade)]) - 1):
            key, subkey = jax.random.split(key)
            w = he_init(subkey, (opt.num_channel["grade"+str(grade)][i], opt.num_channel["grade"+str(grade)][i+1]))
            b = jnp.zeros((opt.num_channel["grade"+str(grade)][i+1], 1))
            params.append((w, b))
        return params

    return model_fn, init_params



# Hessian-vector product without forming full Hessian
def hvp(loss_fn, params, x, y, v):
    flat_params, unflatten = flatten_util.ravel_pytree(params)
    
    def flat_loss(p):
        return loss_fn(unflatten(p), x, y)

    grad_fn = grad(flat_loss)
    return jvp(grad_fn, (flat_params,), (v,))[1]


# Lanczos to compute top/bottom-k eigenvalues of Hessian
def lanczos_eigs(loss_fn, params, x, y, k=10, max_iter=30, largest=True, seed=0):
    flat_params, _ = flatten_util.ravel_pytree(params)
    dim = flat_params.shape[0]
    key = random.PRNGKey(seed)
    q = random.normal(key, (dim,))
    q = q / jnp.linalg.norm(q)

    Q, alphas, betas = [], [], []
    beta = 0.0

    for i in range(max_iter):
        Q.append(q)
        v = hvp(loss_fn, params, x, y, q)
        if i > 0:
            v = v - beta * Q[-2]
        alpha = jnp.dot(q, v)
        alphas.append(alpha)
        v = v - alpha * q
        beta = jnp.linalg.norm(v)
        if beta < 1e-6:
            break
        betas.append(beta)
        q = v / beta

    alphas_arr = jnp.array(alphas)
    betas_arr = jnp.array(betas)
    
    T = jnp.diag(alphas_arr)
    if betas_arr.size >= 1:
        T += jnp.diag(betas_arr[:-1], 1) + jnp.diag(betas_arr[:-1], -1)

    eigvals = jnp.linalg.eigvalsh(T)
    return jnp.sort(eigvals)[-k:] if largest else jnp.sort(eigvals)[:k]





#define loss and gradient
def make_loss_fn(model_fn, acc_y):
    @jit
    def loss_fn(params, x, y):
        logits = model_fn(params, x)[0]                     # shape [num_classes, num_samples]
        logits = logits.T
        y = y.T
        loss = jnp.mean(jnp.sum((logits - y) ** 2, axis=1))           
        return loss
    
    grad_fn = jit(lambda params, x, y: grad(loss_fn)(params, x, y))
    return loss_fn, grad_fn

# Train model with given hyperparameters and data
def train_model(opt, train_data, val_data, train_acc_y, val_acc_y, train_Y, val_Y, normalize, grade):

    key = random.PRNGKey(0)
    
    s_time = time.time() 
    model_fn, init_params = create_network(opt, grade)
    model_pred = jit(lambda params, x : model_fn(params, x)[0])   
    model_loss, model_grad_loss = make_loss_fn(model_fn, train_acc_y)
    model_val_loss, _ = make_loss_fn(model_fn, val_acc_y)
    
    opt_init, opt_update, get_params = optimizers.sgd(opt.learning_rate)

    opt_update = jit(opt_update)


    params = init_params()

    opt_state = opt_init(params)
    train_loss = []
    val_loss = []
    val_loss_smooth = []

    xs = []
    LAR_eigs = []
    SMA_eigs = []

    for i in tqdm(range(opt.epoch['grade'+str(grade)])):

        if opt.eig:
            
            if i%opt.interval['grade'+str(grade)]==0:
                params = get_params(opt_state)
                smallest_eigs = lanczos_eigs(model_loss, params, train_data[0], train_data[1], k=10, largest=False)
                largest_eigs = lanczos_eigs(model_loss, params, train_data[0], train_data[1], k=10, largest=True)
                iter_largest = 1 - opt.learning_rate * smallest_eigs
                iter_smallest = 1 - opt.learning_rate * largest_eigs
                LAR_eigs.append(iter_largest)
                SMA_eigs.append(iter_smallest)
                print(f"Epoch {i}: iter_largest: {iter_largest}")
                print(f"Epoch {i}: iter_smallest: {iter_smallest}")
                    

        opt_state = opt_update(i, model_grad_loss(get_params(opt_state), train_data[0], train_data[1]), opt_state)

        if i % opt.loss_record == 0:
            train_loss.append( normalize * normalize * model_loss(get_params(opt_state), *train_data) )
            val_loss.append( normalize * normalize * model_val_loss(get_params(opt_state), *val_data) )
            xs.append(i)
                    
    e_time = time.time()
    
    train_pred = model_pred(get_params(opt_state), train_data[0])
    val_pred = model_pred(get_params(opt_state), val_data[0])

    _, train_features = model_fn(get_params(opt_state), train_data[0])
    _, val_features = model_fn(get_params(opt_state), val_data[0])

    train_acc_y += normalize * train_pred
    val_acc_y += normalize * val_pred

    normalize = jnp.sqrt(model_loss(get_params(opt_state), *train_data))

    res_train_y = (train_Y - train_acc_y)/normalize
    res_val_y = (val_Y - val_acc_y)/normalize


    if opt.eig:
        history =  {
            'params': get_params(opt_state), 
            'xs': xs,
            'train_pred': train_pred,
            'val_pred': val_pred,
            'train_features': train_features,
            'val_features': val_features,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'train_acc_y': train_acc_y,
            'val_acc_y': val_acc_y,
            'res_train_y': res_train_y,
            'res_val_y': res_val_y,
            'normalize': normalize,
            'time': e_time - s_time,
            'LAR_eigs': LAR_eigs,
            'SMA_eigs': SMA_eigs
        }
    else:
        history =  {
            'params': get_params(opt_state), 
            'xs': xs,
            'train_pred': train_pred,
            'val_pred': val_pred,
            'train_features': train_features,
            'val_features': val_features,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'train_acc_y': train_acc_y,
            'val_acc_y': val_acc_y,
            'res_train_y': res_train_y,
            'res_val_y': res_val_y,            
            'normalize': normalize,
            'time': e_time - s_time
        }        

    return history



def MGDLmodel(opt, data):
    
    
    train_features = data["train_X"]
    val_features = data["val_X"]

    train_Y = data["train_Y"]
    val_Y = data["val_Y"]
    
    res_train_y = train_Y
    res_val_y = val_Y


    train_acc_y = jnp.zeros_like(data["train_Y"])
    val_acc_y = jnp.zeros_like(data["val_Y"])

    SaveHistory = {}

    normalize = 1


    for grade in range(1, opt.grade+1):
        
        input_shape_x = jnp.shape(train_features)[1:]
        
        train_data = [train_features, res_train_y]
        val_data = [val_features, res_val_y]
        
        s_time = time.time() 
        history = train_model(opt, train_data, val_data, train_acc_y, val_acc_y, train_Y, val_Y, normalize, grade)
        e_time = time.time()
        
        train_features = history['train_features']
        val_features = history['val_features']

        train_acc_y = history['train_acc_y']
        val_acc_y = history['val_acc_y']

        res_train_y = history['res_train_y']
        res_val_y = history['res_val_y']

        normalize = history['normalize']


        if opt.eig:
            SaveHistory['grade'+str(grade)] = {
                'params': history['params'],
                'train_loss': history['train_loss'],
                'val_loss': history['val_loss'],
                'LAR_eigs': history['LAR_eigs'],
                'SMA_eigs': history['SMA_eigs'],
                'train_acc_y': train_acc_y,
                'val_acc_y': val_acc_y,
                'res_train_y': res_train_y,
                'res_val_y': res_val_y,
                'normalize': normalize,
                'xs': history['xs'],
                'time': e_time - s_time
            }
        else:
            SaveHistory['grade'+str(grade)] = {
                'params': history['params'],
                'train_loss': history['train_loss'],
                'val_loss': history['val_loss'],
                'train_acc_y': train_acc_y,
                'val_acc_y': val_acc_y,
                'res_train_y': res_train_y,
                'res_val_y': res_val_y,                
                'normalize': normalize,
                'xs': history['xs'],
                'time': e_time - s_time
            }
            
        
        print(f"At grade {grade}, train time: {e_time - s_time},  train loss: {history['train_loss'][-1]}, val loss: {history['val_loss'][-1]}\n")
        

        
    picklename = 'results/MGDLacti%s_grade%d_learningrate%.2e_trainloss%.2e_valloss%.2e.pickle' %(
        opt.activation, opt.grade, opt.learning_rate, history['train_loss'][-1], history['val_loss'][-1]
        )
    
    with open(picklename, 'wb') as f:
        pickle.dump([SaveHistory, opt], f)            



def compute_accuracy(logits, labels):
    """
    logits: shape (num_classes, N)
    labels: shape (num_classes, N) — one-hot format
    """
    pred_class = jnp.argmax(logits, axis=0)
    true_class = jnp.argmax(labels, axis=0)
    accuracy = jnp.mean(pred_class == true_class)
    return accuracy

def analysis(filepath):

    with open(filepath, 'rb') as f:
        [SaveHistory, opt] = pickle.load(f)


    data, _ = data_setup(opt)   


    nshow = 10

    time = 0

    ite = 0 
    # plt.figure(figsize=(10, 6))
    for grade in range(1, opt.grade+1):
        
        history = SaveHistory['grade'+str(grade)]
        xs_record = [x + ite for x in history['xs']]
        ite = ite + history['xs'][-1]
        time = time + history['time'] 

        if grade==1:
            plt.plot(xs_record[1:], history['train_loss'][1:], color="tab:blue", label="Training loss")
        else:
            plt.plot(xs_record[1:], history['train_loss'][1:], color="tab:blue")

        acc = compute_accuracy(history['train_acc_y'], data['train_Y'])
        print(f"Accuracy: {acc * 100:.2f}%")
        print(f"at grade {grade}, train time: {time}, train loss: {history['train_loss'][-1]} val loss: {history['val_loss'][-1]}, acc: {acc}")
            
    plt.xlabel('Epochs', fontsize=20)
    plt.ylabel('Loss', fontsize=20)
    plt.title(f"Multi-Grade", fontsize=20)
    plt.yscale('log')
    plt.ylim([2.0e-3, 1e0])
    # plt.legend(fontsize=20)             
    plt.xticks([0, 3e5, 13e5, 33e5, 53e5], fontsize=18)
    plt.xticks(rotation=45)
    plt.yticks(fontsize=20)
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    fig_filename = f'Fig/MultiGrade_CIFAR10_lr{opt.learning_rate}_Loss.png'
    plt.savefig(fig_filename, format='png', bbox_inches='tight')
    plt.show()    




    if opt.eig:
        sIter=0
        # plt.figure(figsize=(10, 6))
        start=0
        for grade in range(1, opt.grade+1):
            
            history = SaveHistory['grade'+str(grade)]
    
            lenIter = history['xs'][-1] 
    
            Eig_dict_MAX = {}
            for j in range(nshow):
                Eig_dict_MAX['Index'+str(j)] = []
        
            for j in range(nshow):
                for i in range(start, len(history["LAR_eigs"])):
                    Eig_dict_MAX['Index'+str(j)].append(history["LAR_eigs"][i][j])
    
            Eig_dict_MIN = {}
            for j in range(nshow):
                Eig_dict_MIN['Index'+str(j)] = []
        
            for j in range(nshow):
                for i in range(start, len(history["SMA_eigs"])):
                    Eig_dict_MIN['Index'+str(j)].append(history["SMA_eigs"][i][len(history["SMA_eigs"][i])-nshow+j])
            
            for j in range(nshow):
                nlen = len(Eig_dict_MIN['Index'+str(nshow-j-1)])
                if grade==1:
                    plt.plot(range(sIter, sIter+nlen*opt.interval['grade'+str(grade)], opt.interval['grade'+str(grade)]), jnp.array(Eig_dict_MIN['Index'+str(nshow-j-1)]).T, label=f"Index {j}")
                else:
                    plt.plot(range(sIter, sIter+nlen*opt.interval['grade'+str(grade)], opt.interval['grade'+str(grade)]), jnp.array(Eig_dict_MIN['Index'+str(nshow-j-1)]).T)
                  
    
            for j in range(nshow):

                nlen = len(Eig_dict_MAX['Index'+str(nshow-j-1)])
                if grade==1:
                    plt.plot(range(sIter, sIter+nlen*opt.interval['grade'+str(grade)], opt.interval['grade'+str(grade)]), jnp.array(Eig_dict_MAX['Index'+str(nshow-j-1)]).T, linestyle="--", label=f"Index {j+nshow}")
                else:
                    plt.plot(range(sIter, sIter+nlen*opt.interval['grade'+str(grade)], opt.interval['grade'+str(grade)]), jnp.array(Eig_dict_MAX['Index'+str(nshow-j-1)]).T, linestyle="--")
                    
            sIter += opt.epoch['grade'+str(grade)]
            print(f"sIter: {sIter}")
    
    
        plt.plot(range(0, sIter+opt.interval['grade'+str(grade)], opt.interval['grade'+str(grade)]), 1 * jnp.ones_like(jnp.array(range(0, sIter+opt.interval['grade'+str(grade)], opt.interval['grade'+str(grade)]))), 'r--')
        plt.plot(range(0, sIter+opt.interval['grade'+str(grade)], opt.interval['grade'+str(grade)]), -1 * jnp.ones_like(jnp.array(range(0, sIter+opt.interval['grade'+str(grade)], opt.interval['grade'+str(grade)]))), 'r--')
        plt.xlabel('Epochs', fontsize=20)
        plt.ylabel('Eigenvalues', fontsize=20)
        plt.ylim([-4, 1.2])
    
        plt.legend(fontsize=9, loc='upper left', bbox_to_anchor=(1, 1))
        plt.xticks([0, 3e5, 13e5, 33e5, 53e5], fontsize=18)
        plt.xticks(rotation=45)  # Rotate labels 45 degrees 
        plt.yticks(fontsize=20)
        plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
        plt.title('Multi-Grade: Eigenvalues of $\mathbf{I} - \eta\mathbf{H}_{\mathcal{L}}(\mathbf{W}^k)$', fontsize=17)
        fig_filename = f'Fig/MultiGrade_CIFAR10_lr{opt.learning_rate}_Eig.png'
        plt.savefig(fig_filename, format='png', bbox_inches='tight')
        plt.show()


    if opt.eig:
        sIter=0
        # plt.figure(figsize=(10, 6))
        start=0
        for grade in range(1, opt.grade+1):
            
            history = SaveHistory['grade'+str(grade)]
    
            lenIter = history['xs'][-1] 
    
            Eig_dict_MAX = {}
            for j in range(nshow):
                Eig_dict_MAX['Index'+str(j)] = []
        
            for j in range(nshow):
                for i in range(start, len(history["LAR_eigs"])):
                    Eig_dict_MAX['Index'+str(j)].append(history["LAR_eigs"][i][j])
    
            Eig_dict_MIN = {}
            for j in range(nshow):
                Eig_dict_MIN['Index'+str(j)] = []
        
            for j in range(nshow):
                for i in range(start, len(history["SMA_eigs"])):
                    Eig_dict_MIN['Index'+str(j)].append(history["SMA_eigs"][i][len(history["SMA_eigs"][i])-nshow+j])

            for j in range(nshow):
                nlen = len(Eig_dict_MIN['Index'+str(nshow-j-1)])
                if grade==1:
                    plt.plot(range(sIter, sIter+nlen*opt.interval['grade'+str(grade)], opt.interval['grade'+str(grade)]), jnp.array(Eig_dict_MIN['Index'+str(nshow-j-1)]).T, label=f"Index {j}")
                else:
                    plt.plot(range(sIter, sIter+nlen*opt.interval['grade'+str(grade)], opt.interval['grade'+str(grade)]), jnp.array(Eig_dict_MIN['Index'+str(nshow-j-1)]).T)
                  
    
            for j in range(nshow):

                nlen = len(Eig_dict_MAX['Index'+str(nshow-j-1)])
                if grade==1:
                    plt.plot(range(sIter, sIter+nlen*opt.interval['grade'+str(grade)], opt.interval['grade'+str(grade)]), jnp.array(Eig_dict_MAX['Index'+str(nshow-j-1)]).T, linestyle="--", label=f"Index {j+nshow}")
                else:
                    plt.plot(range(sIter, sIter+nlen*opt.interval['grade'+str(grade)], opt.interval['grade'+str(grade)]), jnp.array(Eig_dict_MAX['Index'+str(nshow-j-1)]).T, linestyle="--")
                    
            sIter += opt.epoch['grade'+str(grade)]
            print(f"sIter: {sIter}")
    
    
            plt.xlabel('Epochs', fontsize=20)
            plt.ylabel('Eigenvalues', fontsize=20)
            # plt.ylim([-2, 1.2])
        
            plt.legend(fontsize=9, loc='upper left', bbox_to_anchor=(1, 1))
            plt.xticks(fontsize=20)
            plt.yticks(fontsize=20)
            plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
            plt.title('Multi-Grade: Eigenvalues of $\mathbf{I} - \eta\mathbf{H}_{\mathcal{L}}(\mathbf{W}^k)$', fontsize=17)
            plt.show()

    
    
