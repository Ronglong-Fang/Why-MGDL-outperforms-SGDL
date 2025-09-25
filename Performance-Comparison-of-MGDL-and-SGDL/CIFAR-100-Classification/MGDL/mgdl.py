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

def create_network(grade):
    # Define feature extractor
    if grade==1 or grade==2:
        feature_init, feature_apply = conv_block(64, 3)
        classifier_init, classifier_apply = stax.serial(
            stax.Flatten,
            stax.Dense(64, W_init=he_normal),
            stax.Relu,
            stax.Dense(100, W_init=he_normal)
        )
    else:
        feature_init, feature_apply = conv_block(128, 3)
        classifier_init, classifier_apply = stax.serial(
            stax.Flatten,
            stax.Dense(128, W_init=he_normal),
            stax.Relu,
            stax.Dense(100, W_init=he_normal)
        )

    def init_params(grade, rng_key=random.PRNGKey(0)):
        rng1, rng2 = random.split(rng_key)
        shapes = {
            1: (-1, 32, 32, 3),
            2: (-1, 16, 16, 64),
            3: (-1, 8, 8, 64),
            4: (-1, 4, 4, 128),
        }

        input_shape = shapes[grade]

        shape_after_feature, params_feature = feature_init(rng1, input_shape)
        _, params_classifier = classifier_init(rng2, shape_after_feature)

        # keep them together as one params object (tuple)
        params = (params_feature, params_classifier)
        return params

    def model_fn(params, inputs):
        # inputs: numpy array (we will convert to jnp inside if needed)
        params_feature, params_classifier = params
        # ensure inputs are jnp arrays for JAX ops
        x = jnp.asarray(inputs)
        features = feature_apply(params_feature, x)   # (batch, h, w, c)
        logits = classifier_apply(params_classifier, features)  # (batch,100)
        return logits

    # Apply function to only get features
    def model_feature_fn(params, inputs):
        params_feature, _ = params
        return feature_apply(params_feature, inputs)

    return model_fn, init_params, model_feature_fn





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
def train_model(opt, train_data, val_data, train_acc_y, val_acc_y, train_Y, val_Y, normalize, grade):
    
    s_time = time.time() 
    model_fn, init_params, model_feature_fn = create_network(grade)
    model_pred = jit(lambda params, x : model_fn(params, x))   
    model_loss, model_grad_loss = make_loss_fn(model_fn)
    
    opt_init, opt_update, get_params = optimizers.adam(opt.learning_rate)

    opt_update = jit(opt_update)


    params = init_params(grade)

    opt_state = opt_init(params)
    train_loss = []
    val_loss = []
    val_loss_smooth = []

    xs = []
    LAR_eigs = []
    SMA_eigs = []

    seed = 42
    rng = random.PRNGKey(seed)

    for epoch in tqdm(range(opt.epoch['grade'+str(grade)])):

        if opt.eig and epoch%opt.interval['grade'+str(grade)]==0:
            params = get_params(opt_state)
            smallest_eigs = lanczos_eigs(model_loss, params, train_data[0], train_data[1], k=10, largest=False)
            largest_eigs = lanczos_eigs(model_loss, params, train_data[0], train_data[1], k=10, largest=True)
            iter_largest = 1 - opt.learning_rate * smallest_eigs
            iter_smallest = 1 - opt.learning_rate * largest_eigs
            LAR_eigs.append(iter_largest)
            SMA_eigs.append(iter_smallest)
            print(f"Epoch {epoch}: iter_largest: {iter_largest}")
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
            train_loss.append( normalize * normalize * model_loss(params, *train_data) )
            val_loss.append( normalize * normalize * model_loss(params, *val_data) )
            xs.append(epoch)
                    
    e_time = time.time()
    
    params = get_params(opt_state)  
    train_pred = model_pred(params, train_data[0])
    val_pred = model_pred(params, val_data[0])

    train_features = model_feature_fn(params, train_data[0])
    val_features = model_feature_fn(params, val_data[0])

    train_acc_y += normalize * train_pred
    val_acc_y += normalize * val_pred

    normalize = jnp.sqrt(model_loss(get_params(opt_state), *train_data))

    res_train_y = (train_Y - train_acc_y)/normalize
    res_val_y = (val_Y - val_acc_y)/normalize

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

    if opt.eig:
        history['LAR_eigs'] = LAR_eigs
        history['SMA_eigs'] = SMA_eigs
          

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

        SaveHistory['grade'+str(grade)] = {
            'params': history['params'],
            'train_loss': history['train_loss'],
            'val_loss': history['val_loss'],
            'train_acc_y': train_acc_y,
            'val_acc_y': val_acc_y,                
            'normalize': normalize,
            'xs': history['xs'],
            'time': e_time - s_time
        }
            

        if opt.eig:
            SaveHistory['grade'+str(grade)]['LAR_eigs'] = history['LAR_eigs']
            SaveHistory['grade'+str(grade)]['SMA_eigs'] = history['SMA_eigs']
            
            
            
        
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
    pred_class = jnp.argmax(logits, axis=1)
    true_class = jnp.argmax(labels, axis=1)
    accuracy = jnp.mean(pred_class == true_class)
    return accuracy

def analysis(filepath):

    with open(filepath, 'rb') as f:
        [SaveHistory, opt] = pickle.load(f)


    data, _ = data_setup(opt)   


    nshow = 10

    time = 0

    ite = 0 
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
    
        plt.xlabel('Epochs', fontsize=20)
        plt.ylabel('Eigenvalues', fontsize=20)
        plt.ylim([-1.2, 1.2])
    
        plt.legend(fontsize=9, loc='upper left', bbox_to_anchor=(1, 1))
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
            plt.ylim([-1.2, 1.2])
        
            plt.legend(fontsize=9, loc='upper left', bbox_to_anchor=(1, 1))
            plt.xticks(fontsize=20)
            plt.yticks(fontsize=20)
            plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
            plt.title('Multi-Grade: Eigenvalues of $\mathbf{I} - \eta\mathbf{H}_{\mathcal{L}}(\mathbf{W}^k)$', fontsize=17)
            plt.show()

    
    
