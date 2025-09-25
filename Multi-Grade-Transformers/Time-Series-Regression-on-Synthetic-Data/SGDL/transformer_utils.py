"""
Single-Grade Transformer for Time-Series Regression in JAX/Flax
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
    

    plt.figure(figsize=(15, 4))
    plt.plot(y)
    plt.show()
    
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
    dropout: float = 0.0 
    
    @nn.compact
    def __call__(self, x, train=True):
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
        x = nn.LayerNorm()(x)
        x = jnp.mean(x, axis=1)
        y = nn.Dense(1)(x)
        return y.squeeze(-1)


# -------------------------------
# Training
# -------------------------------

class TrainState(train_state.TrainState):
    pass

def mse_loss(pred, target):
    return jnp.mean((pred - target)**2)

# @jax.jit
def train_step(state, X, y, rng):
    def loss_fn(params):
        y_hat = state.apply_fn({'params': params}, X, train=True, rngs={'dropout': rng})
        loss = mse_loss(y_hat, y)
        return loss
    grads = jax.grad(loss_fn)(state.params)
    state = state.apply_gradients(grads=grads)
    return state


def predict_in_batches(model, params, X, batch_size=128):
    y_hat_list = []
    for i in range(0, len(X), batch_size):
        X_batch = jnp.array(X[i:i+batch_size])
        y_hat_batch = model.apply({'params': params}, X_batch, train=False)
        y_hat_list.append(np.array(y_hat_batch))
    return np.concatenate(y_hat_list)


def train_model(opt):
    rng = jax.random.PRNGKey(opt.seed)
    np.random.seed(opt.seed)

    y = make_sine_dataset(opt.n_total, noise_std=opt.noise_std, seed=opt.seed)
    X, Z = build_windows(y, opt.T)
    n_train = int(0.8 * len(X))
    X_train, y_train = X[:n_train], Z[:n_train]
    X_test, y_test = X[n_train:], Z[n_train:]

    model = SingleGradeTransformer(
        d_model=opt.d_model,
        n_layers=opt.n_layers,
        n_heads=opt.n_heads,
        d_ff=opt.d_ff,
        dropout=opt.dropout
    )


    print("Before model.init")
    init_vars = model.init(rng, jnp.zeros((1, opt.T), dtype=jnp.float32), train=True)
    print("After model.init")
    
    optimizer = optax.adam(opt.lr)
    state = TrainState.create(apply_fn=model.apply, params=init_vars['params'], tx=optimizer)

    Train_MSE = []
    epoch_step = []
    s_time = time.time() 
    for epoch in tqdm(range(opt.epochs)):
        perm = np.random.permutation(len(X_train))
        for i in range(0, len(X_train), opt.batch_size):
            idx = perm[i:i+opt.batch_size]
            rng, subkey = jax.random.split(rng)
            state = train_step(state, jnp.array(X_train[idx]), jnp.array(y_train[idx]), subkey)

        if (epoch+1) % opt.loss_record == 0:
            y_hat_train_batch = predict_in_batches(model, state.params, X_train, batch_size=opt.batch_size)
            train_mse = float(mse_loss(jnp.array(y_hat_train_batch), jnp.array(y_train)))
            Train_MSE.append(train_mse)
            epoch_step.append(epoch+1)

    e_time = time.time()
    y_hat_train = predict_in_batches(model, state.params, X_train, batch_size=opt.batch_size)
    y_hat_test = predict_in_batches(model, state.params, X_test, batch_size=opt.batch_size)
    Test_MSE = float(mse_loss(y_hat_test, jnp.array(y_test)))

    history = {
        'Train_MSE': Train_MSE,
        'Test_MSE': Test_MSE,
        'y_hat_train': y_hat_train,
        'y_hat_test': y_hat_test,
        'y': y,
        'epoch_step': epoch_step,
        'n_train': n_train,
        'time': e_time - s_time
    }

    picklename = f'results/SGDLTransformer_epoch{opt.epochs}_lr{opt.lr:.2e}_TrMse{Train_MSE[-1]:.2e}_TeMse{Test_MSE:.2e}.pickle'
    with open(picklename, 'wb') as f:
        pickle.dump([history, opt], f)

    return 


def analysis(filepath):
    """
    -----------
    history : dict
        
        - 'Train_MSE'
        - 'y_true_test'
        - 'y_hat_test'
    figsize : tuple
    """

    with open(filepath, 'rb') as f:
        [history, opt] = pickle.load(f)

    
    print(f"TrMSE: {history['Train_MSE'][-1]}, TeMSE: {history['Test_MSE']}, Time: {history['time']}")
    
    plt.plot(history['epoch_step'], history['Train_MSE'])
    plt.title("Single-Grade", fontsize=20)
    plt.xlabel("Epochs", fontsize=20)
    plt.ylabel("MSE", fontsize=20)
    plt.ylim([1e-2, 1e0])
    plt.yscale("log")
    plt.show()

    y_true = np.array(history['y'])
    y_hat_train = np.array(history['y_hat_train'])
    y_hat_test = np.array(history['y_hat_test'])

    
    print(f"n_test: {opt.n_total - history['n_train']}, {len(y_hat_test)}")
    plt.plot(range(history['n_train']+opt.T+1, opt.n_total), y_true[history['n_train']+opt.T+1:opt.n_total], label='True', color='tab:red')
    plt.plot(range(history['n_train']+opt.T+1, opt.n_total), y_hat_test, label='Test Predict', color='tab:green')
    plt.title(f"SGT", fontsize=20)
    plt.xlabel("Time step", fontsize=20)
    plt.ylabel("Value", fontsize=20)
    plt.legend(fontsize=15)
    plt.show()

    plt.plot(y_true, label='True', color='tab:red')
    plt.plot(range(opt.T, history['n_train']+opt.T), y_hat_train, label='Train Predict', color='tab:blue')
    plt.plot(range(history['n_train']+opt.T+1, opt.n_total), y_hat_test, label='Test Predict', color='tab:green')
    plt.title(f"SGT", fontsize=20)
    plt.xlabel("Time step", fontsize=20)
    plt.ylabel("Value", fontsize=20)
    plt.legend(fontsize=15)
    plt.show()


