import numpy
import jax
import jax.numpy as jnp
from jax import jit, grad, random, lax
from jax.example_libraries import stax, optimizers
import pickle
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import os, imageio
from jax.scipy.signal import convolve
from tensorflow.keras.datasets import cifar100  
import optax

from jax import flatten_util, jvp
from jax.nn import one_hot


def data_setup(opt):

    # Load data
    (train_X, train_Y), (test_X, test_Y) = cifar100.load_data()

    num_classes = 100  # CIFAR-100类别数
    
    train_Y = one_hot(train_Y.squeeze(), num_classes)  # .squeeze() 去除多余维度
    test_Y = one_hot(test_Y.squeeze(), num_classes)

    # Use fixed random key for reproducibility
    rand_key = random.PRNGKey(0)
    
    # Limit to 10000 training samples (randomly selected)
    num_samples = 10000
    indices = random.choice(rand_key, len(train_X), shape=(num_samples,), replace=False)
    
    train_X = train_X[indices]
    train_Y = train_Y[indices]
    
    train_X = train_X.astype('float32') / 255.0
    test_X = test_X.astype('float32') / 255.0

    train_Y = train_Y.reshape(train_Y.shape[0], -1)
    test_Y = test_Y.reshape(test_Y.shape[0], -1)

    print(f"train_X shape: {train_X.shape}, dtype: {train_X.dtype}")
    print(f"test_X shape: {test_X.shape}, dtype: {test_X.dtype}")

    print(f"train_Y shape: {train_Y.shape}, dtype: {train_Y.dtype}")
    print(f"test_Y shape: {test_Y.dtype}, dtype: {train_Y.dtype}")

    data = {}
    data['train_X'] = train_X  # shape (10000, 32, 32, 3)
    data['train_Y'] = train_Y  # shape (10000, 100)
    data['val_X'] = test_X  # shape (10000, 32, 32, 3)
    data['val_Y'] = test_Y  # shape (10000, 100)

    opt.ntrain = train_X.shape[0]
    opt.num_train = num_samples

    return data, opt

def he_normal(key, shape, dtype=jnp.float32):
    fan_in = 1
    if len(shape) == 2:
        fan_in = shape[0]
    elif len(shape) == 4:
        receptive_field_size = shape[0] * shape[1]
        fan_in = shape[2] * receptive_field_size
    else:
        fan_in = shape[0]
    std = jnp.sqrt(2.0 / fan_in)
    return random.normal(key, shape, dtype) * std

def conv_block(channels, num_layers):
    layers = []
    for _ in range(num_layers):
        layers += [
            stax.Conv(channels, (3, 3), padding='SAME', W_init=he_normal),
            stax.Relu,
        ]
    layers.append(stax.AvgPool((2, 2), strides=(2, 2)))  # 改成 Average Pool 降采样
    return stax.serial(*layers)

def create_network():

    init_fun, apply_fun = stax.serial(
        conv_block(64, 3),   
        conv_block(64, 3),   
        conv_block(128, 3),  
        conv_block(128, 3), 
        stax.Flatten,
        stax.Dense(128, W_init=he_normal),  
        stax.Relu,
        stax.Dense(100, W_init=he_normal), 
    )
    
    def init_params():
        rng_key = random.PRNGKey(0)
        output_shape, params = init_fun(rng_key, (-1, 32, 32, 3))
        return params

    def model_fn(params, inputs):
        return apply_fun(params, inputs)

    return model_fn, init_params



# Hessian-vector product without forming full Hessian
def hvp(loss_fn, params, x, y, v):
    flat_params, unflatten = flatten_util.ravel_pytree(params)
    
    def flat_loss(p):
        return loss_fn(unflatten(p), x, y)

    grad_fn = grad(flat_loss)
    return jvp(grad_fn, (flat_params,), (v,))[1]




def lanczos_eigs(loss_fn, params, x, y, k=10, max_iter=30, largest=True, seed=0):
    flat_params, _ = flatten_util.ravel_pytree(params)
    dim = flat_params.shape[0]
    key = random.PRNGKey(seed)
    q0 = random.normal(key, (dim,))
    q0 = q0 / jnp.linalg.norm(q0)

    def step(carry, _):
        q, q_prev, alpha_arr, beta_arr, i = carry
        v = hvp(loss_fn, params, x, y, q)
        v = lax.cond(
            i > 0,
            lambda v_: v_ - beta_arr[i-1] * q_prev,
            lambda v_: v_,
            operand=v,
        )
        alpha = jnp.dot(q, v)
        v = v - alpha * q
        beta = jnp.linalg.norm(v)
        beta = jnp.where(beta < 1e-6, 0.0, beta)
        q_new = jnp.where(beta > 0, v / beta, q)

        alpha_arr = alpha_arr.at[i].set(alpha)
        beta_arr = beta_arr.at[i].set(beta)

        return (q_new, q, alpha_arr, beta_arr, i+1), None

    alpha_arr = jnp.zeros(max_iter)
    beta_arr = jnp.zeros(max_iter)
    carry = (q0, jnp.zeros_like(q0), alpha_arr, beta_arr, 0)
    carry, _ = lax.scan(step, carry, None, length=max_iter)

    alpha_arr = carry[2]
    beta_arr = carry[3]

    T = jnp.diag(alpha_arr)
    T += jnp.diag(beta_arr[:-1], 1) + jnp.diag(beta_arr[:-1], -1)

    eigvals = jnp.linalg.eigvalsh(T)
    return jnp.sort(eigvals)[-k:] if largest else jnp.sort(eigvals)[:k]



def make_loss_fn(model_fn):
    @jit
    def loss_fn(params, x, y):
        logits = model_fn(params, x)  # shape (batch_size, num_classes)
        # y shape: (batch_size, num_classes)
        loss = jnp.mean(jnp.sum((logits - y) ** 2, axis=1))
        return loss
    grad_fn = jit(lambda params, x, y: grad(loss_fn)(params, x, y))
    return loss_fn, grad_fn


# Train model with given hyperparameters and data
def train_model(opt, train_data, val_data):

    s_time = time.time() 
    model_fn, init_params = create_network()
    model_pred = jit(lambda params, x : model_fn(params, x))
    model_loss, model_grad_loss = make_loss_fn(model_fn)
    
    opt_init, opt_update, get_params = optimizers.adam(opt.learning_rate)

    opt_update = jit(opt_update)
            
    params = init_params()
    opt_state = opt_init(params)

    
    train_loss = []
    val_loss = []

    val_loss_smooth = []
    
    xs = []

    LAR_eigs = []
    SMA_eigs = []


    seed = 42
    rng = random.PRNGKey(seed)
    
    for epoch in tqdm(range(opt.epoch)):

        if opt.eig and epoch % opt.interval == 0:
            params = get_params(opt_state)               
            smallest_eigs = lanczos_eigs(model_loss, params, train_data[0], train_data[1], k=10, largest=False)
            largest_eigs = lanczos_eigs(model_loss, params, train_data[0], train_data[1], k=10, largest=True)
            
        
        
            iter_largest = 1 - opt.learning_rate * smallest_eigs
            iter_smallest = 1 - opt.learning_rate * largest_eigs
            LAR_eigs.append(iter_largest)
            SMA_eigs.append(iter_smallest)
            print(f"Epoch {epoch}: itargest: {iter_largest}")
            print(f"Epoch {epoch}: iter_smallest: {iter_smallest}")


        rng, subkey = random.split(rng)
        perm = random.permutation(subkey, opt.num_train)
        train_X_shuffled = train_data[0][perm]
        train_Y_shuffled = train_data[1][perm]
    
        for start in range(0, opt.num_train, opt.batch_size):
            end = start + opt.batch_size
            batch_X = train_X_shuffled[start:end]
            batch_Y = train_Y_shuffled[start:end]
    
            opt_state = opt_update(epoch, model_grad_loss(get_params(opt_state), batch_X, batch_Y), opt_state)


        if epoch % opt.loss_record == 0:
            params = get_params(opt_state)     
            temp_train_loss = model_loss(params, *train_data)
            temp_val_loss = model_loss(params, *val_data)
            train_loss.append(temp_train_loss)
            val_loss.append(temp_val_loss)
            xs.append(epoch)
                


                    
    e_time = time.time()
    
    params = get_params(opt_state)         
    train_pred = model_pred(params, train_data[0])
    val_pred = model_pred(params, val_data[0])
    
    history = {
        'params': params,
        'xs': xs,
        'train_pred': train_pred,
        'val_pred': val_pred,
        'train_loss': train_loss,
        'val_loss': val_loss,
        'train_time': e_time - s_time,
    }

    if opt.eig:
        history['LAR_eigs'] = LAR_eigs
        history['SMA_eigs'] = SMA_eigs

    picklename = 'results/SGDLacti%s_epoch%d_learningrate%.4e_trainloss%.4e_valloss%.4e.pickle' %(
        opt.activation, opt.epoch, opt.learning_rate, history['train_loss'][-1], history['val_loss'][-1]
        )
    
    with open(picklename, 'wb') as f:
        pickle.dump([history, opt], f)  


    return 

def compute_accuracy(logits, labels):
    """
    logits: shape (num_classes, N)
    labels: shape (num_classes, N) — one-hot format
    """
    pred_class = jnp.argmax(logits, axis=1)
    true_class = jnp.argmax(labels, axis=1)
    accuracy = jnp.mean(pred_class == true_class)
    return accuracy

def analysis(filepath):


    with open(filepath, 'rb') as f:
        [history, opt] = pickle.load(f)


    data, _ = data_setup(opt)

    model_fn, _ = create_network()

    model_loss, _ = make_loss_fn(model_fn)
    params = history['params']
    loss = model_loss(params, data['train_X'], data['train_Y'])

    print(f"loss: {loss}")
    acc = compute_accuracy(history['train_pred'], data['train_Y'])
    print(f"Accuracy: {acc * 100:.2f}%")

    
    nshow = 10    
    nn = len(history['xs'])
    start = 0
    end = nn
    plt.plot(history['xs'][start:end], history['train_loss'][start:end], color="tab:blue")  
    plt.xlabel('Epochs', fontsize=20)
    plt.ylabel('Loss', fontsize=20)
    plt.title(f"Single-Grade", fontsize=20)
    plt.yscale('log')

    plt.yticks(fontsize=20)
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    fig_filename = f'Fig/SingleGrade_CIFAR10_lr{opt.learning_rate}_Loss.png'
    plt.savefig(fig_filename, format='png', bbox_inches='tight')
    plt.show()

    print(f"train time: {history['train_time']}, train loss: {history['train_loss'][-1]} val loss: {history['val_loss'][-1]}")


    if opt.eig:
        Eig_dict_MAX = {}
        for j in range(nshow):
            Eig_dict_MAX['Index'+str(j)] = []
    
        for j in range(nshow):
            for i in range(0, len(history["LAR_eigs"])):
                Eig_dict_MAX['Index'+str(j)].append(history["LAR_eigs"][i][j])
        
        Eig_dict_MIN = {}
        for j in range(nshow):
            Eig_dict_MIN['Index'+str(j)] = []
    
        for j in range(nshow):
            for i in range(0, len(history["SMA_eigs"])):
                Eig_dict_MIN['Index'+str(j)].append(history["SMA_eigs"][i][j])
    
        for j in range(nshow):
            nn = len(Eig_dict_MIN['Index'+str(nshow-j-1)])
            y_ones = numpy.ones(nn)
            
            plt.plot(range(0, opt.interval*nn, opt.interval), jnp.array(Eig_dict_MIN['Index'+str(nshow-j-1)]).T, label=f"Index {j}")

    
        for j in range(nshow):
            nn = len(Eig_dict_MAX['Index'+str(nshow-j-1)])
            
            plt.plot(range(0, opt.interval*nn, opt.interval), jnp.array(Eig_dict_MAX['Index'+str(nshow-j-1)]).T, linestyle="--", label=f"Index {j+nshow}")


        plt.xlabel('Epochs', fontsize=20)
        plt.ylabel('Eigenvalues', fontsize=20)
    
        plt.legend(fontsize=9, loc='upper left', bbox_to_anchor=(1, 1))
        plt.yticks(fontsize=20)
        plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
        plt.title('Single-Grade: Eigenvalues of $\mathbf{I} - \eta\mathbf{H}_{\mathcal{L}}(\mathbf{W}^k)$', fontsize=17)
        fig_filename = f'Fig/SingleGrade_CIFAR10_lr{opt.learning_rate}_Eig.png'
        plt.savefig(fig_filename, format='png', bbox_inches='tight')
        plt.show()

