# %%
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

# import torch
import matplotlib.pyplot as plt

import seaborn as sns
sns.set(style="white", context="talk")
plt.rcParams['savefig.facecolor'] = 'white'


# %%
# train_path = "/kaggle/input/ts-forecasting/train.parquet"
# test_path = "/kaggle/input/ts-forecasting/test.parquet"

train_path = "ts-forecasting/train.parquet"
test_path = "ts-forecasting/test.parquet"

df_train = pd.read_parquet(train_path)
df_test = pd.read_parquet(test_path)

# %%
df_train.head()


# %% [markdown]
# ### Visualising the time series for different codes 


# %%
codes = df_train['code'].unique()
subcodes = df_train['sub_code'].unique()
categories = df_train['sub_category'].unique()
horizons = df_train['horizon'].unique()
ts_index = df_train['ts_index'].unique()

print(f"There are {len(codes)} unique codes in the training set.")
print(f"There are {len(subcodes)} unique subcodes in the training set.")
print(f"There are {len(categories)} unique subcategories in the training set.")
print(f"There are {len(horizons)} unique horizons in the training set.")
print(f"There are {len(ts_index)} unique time steps in the training set.")
print(f"The maximal number of ids is : {len(codes)*len(subcodes)*len(categories)*len(horizons)*29}")
print(f"There is {len(df_train)} rows in the training set.")

#%%
def plot_series_2(df,code, subcode, subcategory, horizon, zoom_start=0, zoom_end=1):
    time = df[(df["code"]==code) & (df["sub_code"]==subcode) & (df["sub_category"]==subcategory) & (df["horizon"]==horizon)].ts_index
    values = df[(df["code"]==code) & (df["sub_code"]==subcode) & (df["sub_category"]==subcategory) & (df["horizon"]==horizon)].y_target
    idx = time.argsort() 
    
    fig, ax = plt.subplots(nrows=1, ncols=2,figsize=(20,8))
    ax[0].plot(time.iloc[idx].values, values.iloc[idx].values, label=f"{code} {subcode} {subcategory} {horizon}")

    ax[0].grid()
    ax[0].legend()

    n = len(time)
    ax[1].plot(time.iloc[idx][int(zoom_start*n):int(zoom_end*n)],values.iloc[idx][int(zoom_start*n):int(zoom_end*n)], label=f"Zoomed Time series between index {int(n*zoom_start)} and {int(n*zoom_end)}")
    ax[1].grid()
    ax[1].legend()
    
    plt.show()

    return time,values

# %% [markdown]
# There are many subcodes for each code. This explains why there are bars at different point times. The more the height of the bars is large, the more the subcode series behave differently. Neat work should be done  

# %%
t,v = plot_series_2(df_train, "W2MW3G2L", "J0G2B0KU", "PZ9S1Z4V", 25, zoom_start=0.90, zoom_end=1)

# %%
def plot_feature(df, code, subcode, subcategory, horizon, feature, zoom_start=0, zoom_end=1):
    time = df[(df["code"]==code) & (df["sub_code"]==subcode) & (df["sub_category"]==subcategory) & (df["horizon"]==horizon)].ts_index
    values = df[(df["code"]==code) & (df["sub_code"]==subcode) & (df["sub_category"]==subcategory) & (df["horizon"]==horizon)][feature]
    idx = time.argsort()

    fig, ax = plt.subplots(nrows=1, ncols=2,figsize=(20,8))
    ax[0].plot(time.iloc[idx].values, values.iloc[idx].values, label=f"Feature {feature} for {code} {subcode} {subcategory} {horizon}")

    ax[0].grid()
    ax[0].legend()

    n = len(time)
    ax[1].plot(time.iloc[idx][int(zoom_start*n):int(zoom_end*n)],values.iloc[idx][int(zoom_start*n):int(zoom_end*n)], label=f"Zoomed Time series between index {int(n*zoom_start)} and {int(n*zoom_end)}")
    ax[1].grid()
    ax[1].legend()
    
    plt.show()

    return time,values

# %%
df_train.head()

# %%
t1,v1 = plot_feature(df_train, "W2MW3G2L", "J0G2B0KU", "PZ9S1Z4V", 25, "feature_b", zoom_start=0.5, zoom_end=1)
t2, v2 = plot_feature(df_train, "W2MW3G2L", "J0G2B0KU", "PZ9S1Z4V", 10, "feature_c", zoom_start=0.5, zoom_end=1)



# %%
def remove_ts_suffix(x):
    w = x.split("__")[:-2]
    return "__".join(w)

# %% [markdown]
# There are $\approx 36k$ time series in the training set if we want to consider each tuple code_subcode_sub_cat_horizon as a time serie on its own.

# %%

df_train["short_id"] = df_train["id"].apply(lambda x: remove_ts_suffix(x))
len(df_train["short_id"].unique())
# for code in codes:
#     s, sc, scat = code.split("__")
#         = df_train["id"].apply()
# %%
df_train.head()
# %%
for short_id in df_train["short_id"].unique()[:5]:
    df_sub = df_train[df_train["short_id"]==short_id]
    plt.figure(figsize=(10,5))
    for horizon in df_sub["horizon"].unique():
        df_horizon = df_sub[df_sub["horizon"]==horizon]
        t = df_horizon["ts_index"]
        v = df_horizon["y_target"]
        idx = t.argsort()
        plt.plot(t.iloc[idx].values, v.iloc[idx].values, label=f"Horizon {horizon}")
    plt.title(f"Time series for {short_id}")
    plt.legend()
    plt.grid()
    plt.show()
     
# %%

# For count the number of time steps that have the same short_id and the same horizon

df_train.groupby(["short_id","horizon"]).size().reset_index(name='counts').groupby("horizon")['counts'].max() 
 

# for short_id in df_train["short_id"].unique()[:5]:
#     df_sub = df_train[df_train["short_id"]==short_id]
#     for horizon in df_sub["horizon"].unique():
#         df_horizon = df_sub[df_sub["horizon"]==horizon]
#         t = df_horizon["ts_index"]
#         print(f"Short id {short_id} horizon {horizon} length {len(t)}")
#         max_hor[horizon] = max(max_hor[horizon], len(t))

        
# %%
nb_short_ids = len(df_train["short_id"].unique())
dataset = {}

for short_id in df_train["short_id"].unique()[:2]:
    dfs = []
    for horizon in [1,3,10,25]:
        df_sub = df_train[(df_train["short_id"]==short_id) & (df_train["horizon"]==horizon)]
        
        ts_index = df_sub["ts_index"].values
        idx = ts_index.argsort()
        df_sub = df_sub.iloc[idx]
        # Select from columns column 5 (ts_index) to column -2 (y_target)
        # df_sub = df_sub.iloc[:, 5:-2]

        df_sub = df_sub.iloc[:, 5:]
        
        # Turn this into a numpy array
        arr = df_sub.to_numpy() 
        dfs.append(arr)
        
    dataset[short_id] = dfs
    
# dataset
# %% [markdown]
# Il faudra tout mettre dans un dataloader pour minimiser le temps d'entraînement
# %%


# len(dataset['W2MW3G2L__J0G2B0KU__PZ9S1Z4V'])

dataset['W2MW3G2L__J0G2B0KU__PZ9S1Z4V'][0].shape






#%% Create a complete pipleline for Pytorch dataloading, outputing time series of different lengths, with the same number of features, and the target variable.

import torch
from torch.utils.data import Dataset, DataLoader

class HedgeFundDataset(Dataset):
    def __init__(self, data_path, horizon=1, normalize=False):
        self.df = pd.read_parquet(data_path)

        ## Reorder the rows of the dataframe according to ts_index
        self.df.sort_values(by=["ts_index"], inplace=True)

        self.df["short_id"] = self.df["id"].apply(lambda x: remove_ts_suffix(x))

        ## Get the mean and std of the features for normalization
        feature_cols = [col for col in self.df.columns if col.startswith("feature_")]
        self.feature_colds = feature_cols

        if normalize:
            self.df_features_mins = self.df[feature_cols].min()
            self.df[feature_cols] = self.df[feature_cols] - self.df_features_mins + 1e-6

            ## Tale the log of the features to make them more Gaussian
            self.df[feature_cols] = np.log(self.df[feature_cols])

            self.feature_means = self.df[feature_cols].mean()
            self.feature_stds = self.df[feature_cols].std()

            ## Normalize the features
            self.df[feature_cols] = (self.df[feature_cols] - self.feature_means) / self.feature_stds

        else:
            self.feature_means = None
            self.feature_stds = None
            self.df_features_mins = None

        self.horizon = horizon
        self.short_ids = self.df["short_id"].unique()

    def __len__(self):
        return len(self.short_ids)

    def __getitem__(self, idx):
        short_id = self.short_ids[idx]

        df_sub = self.df[(self.df["short_id"]==short_id) & (self.df["horizon"]==self.horizon)]
        
        ## Implement linear interpolation to fill the missing values of the time series
        # df_sub = df_sub.set_index("ts_index").reindex(range(df_sub["ts_index"].min(), df_sub["ts_index"].max()+1)).interpolate(method="linear").reset_index()

        ## Replace NaNs with mean of the feature for this short_id and this horizon
        # df_sub[self.feature_colds].fillna(self.df[self.feature_colds].mean(), inplace=True)
        

        # print(df_sub["ts_index"])

        # ts_index = df_sub["ts_index"].values
        # indices = ts_index.argsort()
        # df_sub = df_sub.iloc[indices]

        df_sub = df_sub.iloc[:, 5:]
        arr = df_sub.to_numpy(na_value=0.0)

        # mean_datas = df_sub[self.feature_colds].mean().values
        ## Convert nans to means
        # arr = np.nan_to_num(arr, nan=0., posinf=0., neginf=mean_datas)

        X_feats = arr[:, :-2].astype(np.float32)       ## Shape (T, D_x)
        y_target = arr[:, -3:-1].astype(np.float32)    ## Shape (T, D_y)

        # print("Types of Xfeats and y_target:", type(X_feats), type(y_target))

        ## If t
        ts_indices = np.asarray(df_sub["ts_index"].values).astype(np.int32)


        return X_feats, y_target, ts_indices

        ## Return torch tensors
        # return torch.from_numpy(X_feats).float(), torch.from_numpy(y_target).float(), torch.from_numpy(ts_indices).long()

train_dataset = HedgeFundDataset(train_path, horizon=25, normalize=True)

#%%
## Create the dataloader
train_dataloader = DataLoader(train_dataset, 
                              batch_size=1, 
                              shuffle=True, 
                              num_workers=0)

for X, y, ts_ids in train_dataloader:

    print("Types of X, y, ts_ids:", type(X), type(y), type(ts_ids))

    break

#%%

count = 0
feats_to_plot = np.random.choice(train_dataset[0][0].shape[1], size=2, replace=False)
# feats_to_plot = [42, 55]

i, j = feats_to_plot
print(f"Plotting features {i} and {j} for the first time series in the dataset.")
# for X, y, ts_ids in iterator:

# iterator = iter(train_dataset)
# X, y, ts_ids = next(iterator)

for batch in train_dataloader:

    X, y, ts_ids = batch

    print(ts_ids.shape, X.shape, y.shape)

    plt.figure(figsize=(10,5))
    plt.plot(ts_ids[0], X[0, :, i], label=f"Feature {i}") 
    plt.plot(ts_ids[0], X[0, :, j], label=f"Feature {j}")
    plt.plot(ts_ids[0], y[0, :, 0], label="Target variable")
    plt.legend()

    break

# count += 1
# if count == 2:
#     break



# (B, T_max, D_x) -> (B, T, D_y) 
# %%

train_dataset.df.head()

# %%

print(train_dataset.feature_means)

# %%

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import jax
import jax.numpy as jnp
import equinox as eqx
import optax
from jax.tree_util import tree_map
from torch.utils import data

## Jax debug NaNs
jax.config.update("jax_debug_nans", True)


def numpy_collate(batch):
  return tree_map(np.asarray, data.default_collate(batch))

train_dataloader = DataLoader(train_dataset, 
                                batch_size=1, 
                                shuffle=True, 
                                collate_fn=numpy_collate, 
                                num_workers=0)


#%% Setup and Train Transformer


class PositionalEncoding(eqx.Module):
    pe: jax.Array

    def __init__(self, d_model: int, max_len: int = 500):
        pe = jnp.zeros((max_len, d_model))
        position = jnp.arange(0, max_len, dtype=jnp.float32)[:, jnp.newaxis]
        div_term = jnp.exp(jnp.arange(0, d_model, 2, dtype=jnp.float32) * -(jnp.log(10000.0) / d_model))
        
        pe = pe.at[:, 0::2].set(jnp.sin(position * div_term))
        pe = pe.at[:, 1::2].set(jnp.cos(position * div_term))
        self.pe = pe

    def __call__(self, z):
        ## z is of shape (T, D)
        return z + self.pe[:z.shape[0], :]

class TransformerBlock(eqx.Module):
    attention: eqx.nn.MultiheadAttention
    norm1: eqx.nn.LayerNorm
    mlp: eqx.nn.MLP
    norm2: eqx.nn.LayerNorm
    
    def __init__(self, d_model, n_heads, d_ff, dropout, key):
        k1, k2 = jax.random.split(key)
        self.attention = eqx.nn.MultiheadAttention(
            num_heads=n_heads, 
            query_size=d_model, 
            use_query_bias=True, 
            use_key_bias=True, 
            use_value_bias=True, 
            use_output_bias=True, 
            dropout_p=dropout, 
            key=k1
        )               ## input is shaped (T, D) and output is shape (T, D)
        self.norm1 = eqx.nn.LayerNorm(d_model)
        self.mlp = eqx.nn.MLP(
            in_size=d_model, 
            out_size=d_model, 
            width_size=d_ff, 
            depth=1, 
            activation=jax.nn.gelu, 
            key=k2
        )
        self.norm2 = eqx.nn.LayerNorm(d_model)

    def __call__(self, x, mask=None, key=None):
        attn_out = self.attention(x, x, x, mask=mask, key=key)
        x = eqx.filter_vmap(self.norm1)(x + attn_out)
        mlp_out = jax.vmap(self.mlp)(x)                   ## TODO: we don't need MLPs at all ?
        x = eqx.filter_vmap(self.norm2)(x + mlp_out)
        return x


class Transformer(eqx.Module):
    embedding: eqx.nn.Linear
    pos_encoder: PositionalEncoding
    blocks: list
    norm_final: eqx.nn.LayerNorm  # Added final normalization
    output_projection: eqx.nn.Linear
    
    d_model: int = eqx.field(static=True)
    n_layers: int = eqx.field(static=True)
    
    def __init__(self, input_dim, output_dim, d_model, n_heads, n_layers, d_ff, max_len, key):
        self.d_model = d_model
        self.n_layers = n_layers
        
        k_emb, k_out = jax.random.split(key)
        k_layers = jax.random.split(key, n_layers)
        
        self.embedding = eqx.nn.Linear(input_dim, d_model, key=k_emb)
        self.pos_encoder = PositionalEncoding(d_model, max_len)
        
        self.blocks = [
            TransformerBlock(d_model, n_heads, d_ff, dropout=0.0, key=k)
            for k in k_layers
        ]
        
        self.norm_final = eqx.nn.LayerNorm(d_model)
        
        # --- CRITICAL FIX: Zero Initialization for Output Projection ---
        self.output_projection = eqx.nn.Linear(d_model, output_dim, key=k_out)
        
        # Manually set weights and bias to zero
        zeros_w = jnp.zeros_like(self.output_projection.weight)
        zeros_b = jnp.zeros_like(self.output_projection.bias)
        self.output_projection = eqx.tree_at(
            lambda l: (l.weight, l.bias), 
            self.output_projection, 
            (zeros_w, zeros_b)
        )

    def make_causal_mask(self, seq_len):
        idx = jnp.arange(seq_len)
        mask = idx[:, None] >= idx[None, :]
        return mask

    def __call__(self, x, key=None):

        # print("Input to Transformer:", x.shape)  # Debugging statement

        # 1. Embedding scaled by sqrt(d_model) for stability
        x = jax.vmap(self.embedding)(x) * jnp.sqrt(self.d_model)     ## takes in (T, D_in)
        x = self.pos_encoder(x)                                     ## (T, D_model)

        # 2. Transformer Blocks
        for block in self.blocks:
            x = block(x, mask=None)                                 ## (T, D_model)
        
        # 3. Final Norm (Standard in Pre-Norm architectures)
        x = jax.vmap(self.norm_final)(x)                            ## (T, D_model)

        # 5. Predict Delta
        delta = jax.vmap(self.output_projection)(x)                 ## (T, D_out)

        y = delta

        return y

# (T, D_in) -> (T, D_model) -> (T, D_model) ... (T, D_model) -> (T, D_out)


#%% Train the model, and predict and visualize

CONFIG = {
    "seed": 42,
    "num_epochs": 1,
    "print_every": 1
}


model = Transformer(
    input_dim=88,
    output_dim=1,
    d_model=128, ## increase this
    n_heads=4,
    n_layers=2,
    d_ff=128,
    max_len=512,
    key=jax.random.PRNGKey(CONFIG["seed"])
)

#%%
# Test forward pass
test_x = X[0]  # Shape (T, D_x)
test_x = jnp.asarray(test_x)  # Convert to JAX array
predicted_traj = model(test_x, key=jax.random.PRNGKey(1))  # Shape (16, 1)
print("Input shape:", test_x.shape)
print("Output shape:", predicted_traj.shape)



#%%
## Now train the model to minimize MSE over the predicted trajectory

## Fully functional training loop
optimizer = optax.adam(learning_rate=1e-6)
opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))


def _clip01(x):
    return jnp.minimum(jnp.maximum(x, 0.0), 1.0)

def weighted_rmse_score(y_target, y_pred, w):
    denom = jnp.sum(w * y_target ** 2)
    ratio = jnp.sum(w * (y_target - y_pred) ** 2) / denom
    clipped = _clip01(ratio)
    val = 1.0 - clipped
    return jnp.sqrt(val)

def mse(y_target, y_pred):
    return jnp.mean((y_target - y_pred) ** 2)

def loss_fn(model, x, y, w=None):
    y_hat = model(x)
    if w is None:
        return mse(y, y_hat)
    else:
        return -weighted_rmse_score(y, y_hat, w)



# @eqx.filter_jit
def train_step(model, x, y_weights, opt_state):
    x = x[0]  # Remove batch dimension, now x is (T, D_x)
    y, w = y_weights[0, :, 0], y_weights[0, :, 1]

    # loss, grads = eqx.filter_value_and_grad(loss_fn)(model, x, y, w)
    loss, grads = eqx.filter_value_and_grad(loss_fn)(model, x, y, None)

    updaes, opt_state = optimizer.update(grads, opt_state, model)
    model = eqx.apply_updates(model, updaes)

    return model, opt_state, loss


losses = []

for epoch in range(CONFIG["num_epochs"]):

    for batch_id, batch in enumerate(train_dataloader):
        x, y_weights, ts_index = jax.device_put(batch)  # Move batch to device

        model, opt_state, loss = train_step(model, x, y_weights, opt_state)
        losses.append(loss)

        # if ((epoch+1) % CONFIG["print_every"] == 0) and (batch_id % 1==0):
        if ((epoch+1) % CONFIG["print_every"] == 0):
            print(f"Epoch {epoch+1}, Batch {batch_id+1}, Loss: {loss:.4f}")

        if batch_id > 20:  # Limit number of batches for quick testing
            break

#%%
# Plot training loss
plt.figure(figsize=(10, 6))
plt.plot(losses)
plt.title("Training Loss over Steps")
plt.xlabel("Training Steps")
plt.ylabel("MSE Loss")
plt.yscale("log")
plt.show()


#%%

