"""
Multi-Grade Transformer for Time-Series Regression in JAX/Flax
- Task: predict x_{t+1} given past T values (simple sine wave).
- Model: Single grade encoder-only Transformer
- Loss: MSE
"""

from __future__ import annotations
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from flax import linen as nn
from flax.training import train_state
import optax
import pickle
from tqdm import tqdm
import time


# -------------------------------
# Data
# -------------------------------
def make_sine_dataset(n_total=2000, noise_std=0.1, seed=0):
    
    rng = np.random.default_rng(seed)
    t = np.arange(n_total, dtype=np.float32)
    
    # Base signals
    y = np.sin(2 * np.pi * t / 50)  # long period
    y += 0.5 * np.sin(2 * np.pi * t / 23)  # short period
    
    # Frequency modulation
    y += 0.3 * np.sin(2 * np.pi * t / 10 + 0.5 * np.sin(2 * np.pi * t / 100))
    
    # Slowly changing trend
    y += 0.01 * t
    
    return y.astype(np.float32)


def build_windows(y, T):
    X, Z = [], []
    for i in range(len(y) - T - 1):
        X.append(y[i:i+T])
        Z.append(y[i+T])
    return np.stack(X).astype(np.float32), np.stack(Z).astype(np.float32)

# -------------------------------
# Model components
# -------------------------------

class PositionalEncoding(nn.Module):
    d_model: int

    @nn.compact
    def __call__(self, x):
        T = x.shape[1]
        pos = jnp.arange(T)[:, None]
        i = jnp.arange(self.d_model)[None, :]
        angle_rates = 1 / (10000 ** (2 * (i // 2) / self.d_model))
        angles = pos * angle_rates
        pe = jnp.where(i % 2 == 0, jnp.sin(angles), jnp.cos(angles))
        pe = jnp.broadcast_to(pe, x.shape)
        return x + pe

class MLP(nn.Module):
    d_hidden: int
    d_out: int

    @nn.compact
    def __call__(self, x, train=True):
        x = nn.Dense(self.d_hidden)(x)
        x = nn.relu(x)
        x = nn.Dense(self.d_out)(x)
        return x

class TransformerBlock(nn.Module):
    d_model: int
    n_heads: int
    d_ff: int
    dropout: float = 0.0

    @nn.compact
    def __call__(self, x, train=True):
        h = nn.LayerNorm()(x)
        h = nn.SelfAttention(num_heads=self.n_heads, qkv_features=self.d_model, out_features=self.d_model,
                             dropout_rate=self.dropout, deterministic=not train)(h)
        x = x + h
        h = nn.LayerNorm()(x)
        h = MLP(self.d_ff, self.d_model)(h, train=train)
        x = x + h
        
        return x

class SingleGradeTransformer(nn.Module):
    d_model: int
    n_layers: int
    n_heads: int
    d_ff: int
    grade: int
    dropout: float = 0.0 

    @nn.compact
    def __call__(self, x, train=True):
        if self.grade==1:
            x = x[..., None]
            x = nn.Dense(self.d_model)(x)
            x = PositionalEncoding(self.d_model)(x)
            
        for _ in range(self.n_layers):
            x = TransformerBlock(
                d_model=self.d_model,
                n_heads=self.n_heads,
                d_ff=self.d_ff,
                dropout=self.dropout
            )(x, train=train)
        feature = nn.LayerNorm()(x)
        x = jnp.mean(feature, axis=1)
        y = nn.Dense(1)(x)
        return y.squeeze(-1), feature

# -------------------------------
# Training
# -------------------------------

class TrainState(train_state.TrainState):
    pass

def mse_loss(pred, target):
    return jnp.mean((pred - target)**2)

@jax.jit
def train_step(state, X, y, rng):
    def loss_fn(params):
        y_hat, _ = state.apply_fn({'params': params}, X, train=True, rngs={'dropout': rng})
        loss = mse_loss(y_hat, y)
        return loss
    grads = jax.grad(loss_fn)(state.params)
    state = state.apply_gradients(grads=grads)
    return state


def predict_in_batches(model, params, X, batch_size=128):
    y_list = []
    feature_list = []
    for i in range(0, len(X), batch_size):
        X_batch = jnp.array(X[i:i+batch_size])
        y_batch, feature_batch = model.apply({'params': params}, X_batch, train=False)
        y_list.append(np.array(y_batch))
        feature_list.append(np.array(feature_batch))
    return np.concatenate(y_list), np.concatenate(feature_list)


def train_model(opt, train_data, test_data, y_acc_train, y_acc_test, y_train_org, y_test_org, normalize, grade):

    rng = jax.random.PRNGKey(opt.seed)
    np.random.seed(opt.seed)

    model = SingleGradeTransformer(
        d_model=opt.d_model,
        n_layers=opt.n_layers,
        n_heads=opt.n_heads,
        d_ff=opt.d_ff,
        grade=grade,
        dropout=opt.dropout
    )

    if grade == 1:
        dummy_x = jnp.zeros((1, opt.T), dtype=jnp.float32)  # raw input
    else:
        dummy_x = jnp.zeros((1, opt.T, opt.d_model), dtype=jnp.float32)  # input from previous grade

    init_vars = model.init(rng, dummy_x, train=True)


    
    optimizer = optax.adam(opt.lr['grade'+str(grade)])
    state = TrainState.create(apply_fn=model.apply, params=init_vars['params'], tx=optimizer)

    Train_MSE = []
    epoch_step = []

    print(f'*******************grade: {grade}*******************')
    for epoch in tqdm(range(opt.epochs)):
        perm = np.random.permutation(len(train_data[0]))
        for i in range(0, len(train_data[0]), opt.batch_size):
            idx = perm[i:i+opt.batch_size]
            rng, subkey = jax.random.split(rng)
            state = train_step(state, jnp.array(train_data[0][idx]), jnp.array(train_data[1][idx]), subkey)

        if (epoch+1) % opt.loss_record == 0:
            y_train_batch, _ = predict_in_batches(model, state.params, train_data[0], batch_size=opt.batch_size)
            train_mse = normalize * normalize * mse_loss(jnp.array(y_train_batch), jnp.array(train_data[1]))
            Train_MSE.append(train_mse)
            epoch_step.append(epoch+1)

    y_train, feature_train = predict_in_batches(model, state.params, train_data[0], batch_size=opt.batch_size)
    y_test, feature_test = predict_in_batches(model, state.params, test_data[0], batch_size=opt.batch_size)
    Test_MSE = normalize * normalize * mse_loss(y_test, jnp.array(test_data[1]))

    y_acc_train += normalize * y_train
    y_acc_test += normalize * y_test

    normalize = jnp.sqrt(mse_loss(jnp.array(y_train_batch), jnp.array(train_data[1])))

    res_y_train = (y_train_org - y_acc_train)/normalize
    res_y_test = (y_test_org - y_acc_test)/normalize

    history = {
        'Train_MSE': Train_MSE,
        'Test_MSE': Test_MSE,
        'y_acc_train': y_acc_train,
        'y_acc_test': y_acc_test,
        'feature_train': feature_train,
        'feature_test': feature_test,
        'res_y_train': res_y_train,
        'res_y_test': res_y_test,
        'normalize': normalize,
        'epoch_step': epoch_step
    }

    return history


def MGDLmodel(opt):

    rng = jax.random.PRNGKey(opt.seed)
    np.random.seed(opt.seed)

    y = make_sine_dataset(opt.n_total, noise_std=opt.noise_std, seed=opt.seed)
    X, Z = build_windows(y, opt.T)
    n_train = opt.n_train
    X_train_org, y_train_org = X[:n_train], Z[:n_train]
    X_test_org, y_test_org = X[n_train:], Z[n_train:]

    feature_train = X_train_org
    feature_test = X_test_org

    res_y_train = y_train_org
    res_y_test = y_test_org

    y_acc_train = jnp.zeros_like(y_train_org)
    y_acc_test = jnp.zeros_like(y_test_org)

    SaveHistory = {}
    normalize = 1

    for grade in range(1, opt.grade+1):

        print(f"grade: {grade}, shape feature: {jnp.shape(feature_train)}, shape y: {np.shape(res_y_train)}")
        train_data = [feature_train, res_y_train]
        test_data = [feature_test, res_y_test]
        s_time = time.time() 
        history = train_model(opt, train_data, test_data, y_acc_train, y_acc_test, y_train_org, y_test_org, normalize, grade)
        e_time = time.time()

        feature_train = history['feature_train']
        feature_test = history['feature_test']
        res_y_train = history['res_y_train']
        res_y_test = history['res_y_test']

        y_acc_train = history['y_acc_train']
        y_acc_test = history['y_acc_test']

        normalize = history['normalize']
        
    
        SaveHistory['grade'+str(grade)] = {
            'Train_MSE': history['Train_MSE'],
            'Test_MSE': history['Test_MSE'],
            'y_acc_train': y_acc_train,
            'y_acc_test': y_acc_test,                
            'normalize': normalize,
            'epoch_step': history['epoch_step'],
            'time': e_time - s_time
        }
            

    picklename = 'results/MGDL_grade%d_epoch%d_TrMse%.2e_TeMSE%.2e.pickle' %(
        opt.grade, opt.epochs, history['Train_MSE'][-1], history['Test_MSE']
    )
    
    with open(picklename, 'wb') as f:
        pickle.dump([SaveHistory, opt], f)   

    return




def analysis(filepath):

    with open(filepath, 'rb') as f:
        [SaveHistory, opt] = pickle.load(f)

    rng = jax.random.PRNGKey(opt.seed)
    np.random.seed(opt.seed)
    y = make_sine_dataset(opt.n_total, noise_std=opt.noise_std, seed=opt.seed)    
        

    print(opt.lr)

    time = 0

    ite = 0 
    opt.grade=3
    for grade in range(1, opt.grade+1):
        
        history = SaveHistory['grade'+str(grade)]
        epoch_step = [epoch_step + ite for epoch_step in history['epoch_step']]
        ite = ite + history['epoch_step'][-1]
        time = time + history['time'] 

        print(f"grade: {grade}, Train MSE: {history['Train_MSE'][-1]}, Test MSE: {history['Test_MSE']}, time: {history['time']}, total time: {time}")

        if grade==1:
            plt.plot(epoch_step, history['Train_MSE'], color="tab:blue", label="TrMSE")
        else:
            plt.plot(epoch_step, history['Train_MSE'], color="tab:blue")
    
    plt.title("Multi-Grade", fontsize=20)
    plt.xlabel("Epochs", fontsize=20)
    plt.ylabel("MSE", fontsize=20)
    plt.ylim([1e-2, 1e0])
    plt.yscale("log")
    plt.show()

    for grade in range(1, opt.grade+1):
        
        history = SaveHistory['grade'+str(grade)]
        plt.plot(range(opt.n_train+opt.T+1, opt.n_total), y[opt.n_train+opt.T+1:opt.n_total], label='True', color='tab:red')
        plt.plot(range(opt.n_train+opt.T+1, opt.n_total), history['y_acc_test'], label='Test Predict', color='tab:green')
        plt.title(f"MGT", fontsize=20)
        plt.xlabel("Time step", fontsize=20)
        plt.ylabel("Value", fontsize=20)
        plt.legend(fontsize=15)
        plt.show()

        plt.plot(y, label='True', color='tab:red')
        plt.plot(range(opt.T, opt.n_train+opt.T), history['y_acc_train'], label='Train Predict', color='tab:blue')
        plt.plot(range(opt.n_train+opt.T+1, opt.n_total), history['y_acc_test'], label='Test Predict', color='tab:green')
        plt.title(f"MGT", fontsize=20)
        plt.xlabel("Time step", fontsize=20)
        plt.ylabel("Value", fontsize=20)
        plt.legend(fontsize=15)
        plt.show()
