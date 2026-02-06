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
    
dataset
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

        X_feats = arr[:, :-2]       ## Shape (T, D_x)
        y_target = arr[:, -3:-2]    ## Shape (T, D_y)

        # print("Types of Xfeats and y_target:", type(X_feats), type(y_target))

        ## If t

        return X_feats, y_target, df_sub["ts_index"].values

train_dataset = HedgeFundDataset(train_path, horizon=25, normalize=True)
iterator = iter(train_dataset)

#%%
## Create the dataloader
train_dataloader = DataLoader(train_dataset, 
                              batch_size=1, 
                              shuffle=True, 
                              num_workers=0)


#%%

count = 0
feats_to_plot = np.random.choice(train_dataset[0][0].shape[1], size=2, replace=False)
# feats_to_plot = [42, 55]

i, j = feats_to_plot
# for X, y, ts_ids in iterator:

X, y, ts_ids = next(iterator)

# for X, y, ts_ids in train_dataloader:

# print("SHapes:", X.shape, y.shape)
# print("SHapes:", X, y)
## Plot the first few features and the target variable for the first time series in the dataset

print(ts_ids)

plt.figure(figsize=(10,5))
plt.plot(ts_ids, X[:,i], label=f"Feature {i}") 
plt.plot(ts_ids, X[:,j], label=f"Feature {j}")
plt.plot(ts_ids, y[:,0], label="Target variable")
plt.legend()

# count += 1
# if count == 2:
#     break



# (B, T_max, D_x) -> (B, T, D_y) 
# %%

train_dataset.df.head()

# %%

print(train_dataset.feature_means)

# %%
