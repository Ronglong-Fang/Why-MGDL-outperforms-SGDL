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
import pandas as pd

# -------------------------------
# Data
# -------------------------------
def create_data():
    df = pd.read_excel("data/SPX_Index_Daily_Data.xlsx", 
                       sheet_name="Sheet1", names=["Date", "Close"]) 
    df['Date'] = pd.to_datetime(df['Date'])                                                      
    df_after_2000 = df[df['Date'] >= '2000-01-01']
    
    prices_np = df_after_2000['Close'].to_numpy()                                               
    dates_np = df_after_2000['Date'].to_numpy()        
    return prices_np.astype(np.float32), dates_np
    
def build_windows(y, dates, T):
    X, Z, D = [], [], []
    for i in range(len(y) - T - 1):
        X.append(y[i:i+T])        # 历史窗口
        Z.append(y[i+T])          # 预测目标
        d = pd.to_datetime(dates[i+T]).strftime("%Y-%m-%d")  # 转成 pandas.Timestamp 再格式化
        D.append(d)
    return np.stack(X).astype(np.float32), np.stack(Z).astype(np.float32), np.array(D)

def split_data(X, y, test_ratio=0.05, val_ratio=0.05, seed=0):
    num_samples = len(X)
    n_train_val = int(num_samples * (1 - test_ratio))
    n_test = num_samples - n_train_val

    X_train_val = X[:n_train_val]
    y_train_val = y[:n_train_val]
    X_test = X[n_train_val:]
    y_test = y[n_train_val:]
    indices_test = np.arange(n_train_val, num_samples) 

    np.random.seed(seed)
    indices = np.arange(n_train_val)
    np.random.shuffle(indices)

    n_val = int(n_train_val * val_ratio)
    n_train = n_train_val - n_val

    train_idx = indices[:n_train]
    val_idx = indices[n_train:]

    X_train = X_train_val[train_idx]
    y_train = y_train_val[train_idx]
    X_val = X_train_val[val_idx]
    y_val = y_train_val[val_idx]
    
    indices_train = train_idx
    indices_val = val_idx

    return (X_train, y_train, X_val, y_val, X_test, y_test, 
            indices_train, indices_val, indices_test)

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
    dropout: float = 0.0  # 可选默认值

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


    y, date = create_data()
    X, Z, _ = build_windows(y, date, opt.T)

    X_train, y_train, X_val, y_val, X_test, y_test, idx_train, idx_val, idx_test = split_data(X, Z)

    mean = X_train.mean()
    std = X_train.std()

    opt.mean = mean
    opt.std = std
    opt.idx_train = idx_train
    opt.idx_val = idx_val
    opt.idx_test = idx_test
    
    X_train = (X_train - mean) / std
    X_val   = (X_val - mean) / std
    X_test  = (X_test - mean) / std
    
    y_train = (y_train - mean) / std
    y_val   = (y_val - mean) / std
    y_test  = (y_test - mean) / std

    y = (y - mean) / std
    print("n train:", len(X_train))
    print("n val:", len(X_val))
    print("n test:", len(X_test))

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
    Val_MSE = []
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
            y_hat_val_batch = predict_in_batches(model, state.params, X_val, batch_size=opt.batch_size)
            
            train_mse = float(mse_loss(jnp.array(y_hat_train_batch), jnp.array(y_train)))
            val_mse = float(mse_loss(jnp.array(y_hat_val_batch), jnp.array(y_val)))
            Train_MSE.append(train_mse)
            Val_MSE.append(val_mse)
            epoch_step.append(epoch+1)

    e_time = time.time()
    y_hat_train = predict_in_batches(model, state.params, X_train, batch_size=opt.batch_size)
    y_hat_val = predict_in_batches(model, state.params, X_val, batch_size=opt.batch_size)
    y_hat_test = predict_in_batches(model, state.params, X_test, batch_size=opt.batch_size)
    print(f"shape y hat test: {jnp.shape(y_hat_test)}, shape y test: {jnp.shape(y_test)}")
    Test_MSE = float(mse_loss(y_hat_test, jnp.array(y_test)))

    history = {
        'Train_MSE': Train_MSE,
        'Test_MSE': Test_MSE,
        'Val_MSE': Val_MSE,
        'y_hat_train': y_hat_train,
        'y_hat_val': y_hat_val,
        'y_hat_test': y_hat_test,
        'y': y,
        'epoch_step': epoch_step,
        'time': e_time - s_time
    }

    picklename = f'results/SGDLTransformer_epoch{opt.epochs}_lr{opt.lr:.2e}_TrMse{Train_MSE[-1]:.2e}_VaMse{Val_MSE[-1]:.2e}_TeMse{Test_MSE:.2e}.pickle'
    with open(picklename, 'wb') as f:
        pickle.dump([history, opt], f)

    return 


def analysis(filepath, predict):

    with open(filepath, 'rb') as f:
        [history, opt] = pickle.load(f)

    
    print(f"TrMSE: {history['Train_MSE'][-1]}, ValMSE: {history['Val_MSE'][-1]}, TeMSE: {history['Test_MSE']}, Time: {history['time']}")

    plt.plot(history['epoch_step'], history['Train_MSE'], label='TrMSE')
    plt.plot(history['epoch_step'], history['Val_MSE'], label='VaMSE')
    plt.title("Single-Grade", fontsize=20)
    plt.xlabel("Epochs", fontsize=20)
    plt.ylabel("MSE", fontsize=20)
    plt.legend(fontsize=20)
    plt.yscale("log")
    plt.show()

    y_true = np.array(history['y'])
    y_hat_train = np.array(history['y_hat_train'])
    y_hat_val = np.array(history['y_hat_val'])
    y_hat_test = np.array(history['y_hat_test'])
    idx_train = opt.idx_train
    n_train = len(idx_train)
    idx_val = opt.idx_val
    n_val = len(idx_val)
    idx_test = opt.idx_test
    n_test = len(idx_test)
    mean = opt.mean
    std = opt.std

    y_true = y_true*std+mean
    y_hat_train = y_hat_train*std+mean
    y_hat_val = y_hat_val*std+mean
    y_hat_test = y_hat_test*std+mean

    y_hat_train_val_sorted = np.empty(len(idx_train) + len(idx_val))

    y_hat_train_val_sorted[idx_train] = y_hat_train
    y_hat_train_val_sorted[idx_val]  = y_hat_val

    plt.plot(y_true, label='True', color='tab:red')
    plt.plot(range(opt.T, n_train+n_val+opt.T), y_hat_train_val_sorted, label='Train & Val Predict', color='tab:blue')
    plt.plot(range(n_train+n_val+opt.T, n_train+n_val+n_test+opt.T), y_hat_test, label='Test Predict', color='tab:green')
    plt.title(f"SGT", fontsize=20)
    plt.xlabel("Time step", fontsize=20)
    plt.ylabel("Value", fontsize=20)
    plt.legend(fontsize=15)
    plt.show()

    plt.plot(range(n_train+n_val+opt.T, n_train+n_val+n_test+opt.T), y_true[n_train+n_val+opt.T:n_train+n_val+n_test+opt.T], label='True', color='tab:red')
    plt.plot(range(n_train+n_val+opt.T, n_train+n_val+n_test+opt.T), y_hat_test, label='Test Predict', color='tab:green')
    plt.title(f"SGT", fontsize=20)
    plt.xlabel("Time step", fontsize=20)
    plt.ylabel("Value", fontsize=20)
    plt.legend(fontsize=15)
    plt.show()




    y_true = np.array(history['y'])*std + mean
    y_true_afterT = y_true[opt.T+1:]
    y_true_log_return_afterT = np.log(y_true_afterT[1:]/y_true_afterT[:-1])


    y_hat_train_val_test_sorted = np.empty(n_train + n_val + n_test)
    y_hat_train_val_test_sorted[idx_train] = y_hat_train*std + mean
    y_hat_train_val_test_sorted[idx_val] = y_hat_val*std + mean
    y_hat_train_val_test_sorted[idx_test] = y_hat_test*std + mean

    y_pred_log_return = np.log(y_hat_train_val_test_sorted[1:]/y_hat_train_val_test_sorted[:-1])

    rmse_train_val = np.mean((y_true_log_return_afterT[:n_train+n_val] - y_pred_log_return[:n_train+n_val])**2)
    
    rmse_train_val_baseline = np.mean((y_true_log_return_afterT[:n_train+n_val])**2)

    rmse_test = np.mean((y_true_log_return_afterT[n_train+n_val:] - y_pred_log_return[n_train+n_val:])**2)

    rmse_test_baseline = np.mean((y_true_log_return_afterT[n_train+n_val:])**2)
    
    print(f"MSE (train+val): {rmse_train_val:.6f}, baseline: {rmse_train_val_baseline:.6f}")
    print(f"MSE (test): {rmse_test:.6f}, baseline: {rmse_test_baseline:.6f}")

    

    if predict:
        y, date = create_data()
        X, Z, date = build_windows(y, date, opt.T)

        X_train_val_org, y_train_val_org     = X[:n_train+n_val], Z[:n_train+n_val]
        X_test_org, y_test_org               = X[n_train+n_val:], Z[n_train+n_val:]

    
        y_train_val_predict = y_hat_train_val_sorted*std + mean
        y_test_predict = y_hat_test*std + mean
    
    
        date_train_val   = date[:n_train+n_val]
        date_test  = date[n_train+n_val:]

        df_train_val = pd.DataFrame({
            "Date": date_train_val,
            "True": y_train_val_org,
            "Pred": y_train_val_predict
        })
        
        df_test = pd.DataFrame({
            "Date": date_test,
            "True": y_test_org,
            "Pred": y_test_predict
        })
        
        with pd.ExcelWriter("SGDL_SPX_Predictions.xlsx") as writer:
            df_train_val.to_excel(writer, sheet_name="Train & Val", index=False)
            df_test.to_excel(writer, sheet_name="Test", index=False)




