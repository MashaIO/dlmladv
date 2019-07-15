#%%
import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

#%%
np.random.seed(1234)

#%%
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix, recall_score, precision_score
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, LSTM


#%% [markdown]
# # Pre-processing
# ##Read the data - run-to-failure data

#%%
train_df = pd.read_csv('/Users/sheik/repo/course/dlmladv/data/predmaint/PM_train.txt', sep=" ", header=None)
train_df.drop(train_df.columns[[26, 27]], axis=1, inplace=True)
train_df.columns = ['id', 'cycle', 'setting1', 'setting2','setting3', 's1', 's2', 's3', 's4', 's5', 's6', 's7', 's8', 's9', 's10', 's11', 's12', 's13', 's14', 's15', 's16', 's17', 's18', 's19', 's20', 's21']
train_df = train_df.sort_values(['id', 'cycle'])


#%% [markdown]
# ## Read the test data

#%%
test_df = pd.read_csv('/Users/sheik/repo/course/dlmladv/data/predmaint/PM_test.txt', sep=" ", header=None)
test_df.drop(test_df.columns[[26,27]],
    axis=1,
    inplace=True)
test_df.columns = ['id', 'cycle', 'setting1', 'setting2','setting3', 's1', 's2', 's3', 's4', 's5', 's6', 's7', 's8', 's9', 's10', 's11', 's12', 's13', 's14', 's15', 's16', 's17', 's18', 's19', 's20', 's21']
test_df = test_df.sort_values(['id', 'cycle'])


#%% [markdown]
# ## Read ground truth data
# contains the information of true remaining cycles for each engine in the testing data
#%%
truth_df = pd.read_csv('/Users/sheik/repo/course/dlmladv/data/predmaint/PM_truth.txt', sep=" ", header=None)
truth_df.drop(truth_df.columns[[1]], axis=1, inplace=True)



#%% [markdown]
# ## Labeling RUL (remaining useful life)
# label1 to tell if a specific engine is going to fail within w1 cycles or not

#%%
rul = pd.DataFrame(train_df.groupby('id')['cycle'].max()).reset_index()
rul.columns = ['id', 'max']

#%%
train_df = train_df.merge(rul,
    on=['id'],
    how='left')
train_df['RUL'] = train_df['max'] - train_df['cycle']
train_df.drop('max', axis=1, inplace=True)

#%% [markdown]
# Generating label columns for training data (binary classification)
# ** Whether the engine is going to fail within w1 cycles or not? **


#%%
w1 = 30
w0 = 15
train_df['label1'] = np.where(train_df['RUL'] <= w1, 1, 0)

# MinMax normalization (from 0 to 1)
train_df['cycle_norm'] = train_df['cycle']
cols_normalize = train_df.columns.difference(['id', 'cycle', 'RUL', 'label1'])
min_max_scaler = preprocessing.MinMaxScaler()
norm_train_df = pd.DataFrame(min_max_scaler
    .fit_transform(train_df[cols_normalize]),
    columns = cols_normalize,
    index = train_df.index)
join_df = train_df[train_df.columns.
    difference(cols_normalize)].join(norm_train_df)
train_df = join_df.reindex(columns = train_df.columns)
train_df.head()


#%% [markdown]
# ## Preprocessing test dataset

#%%
# MinMax normalization (from 0 to 1)
test_df['cycle_norm'] = test_df['cycle']
norm_test_df = pd.DataFrame(min_max_scaler
    .fit_transform(test_df[cols_normalize]),
    columns = cols_normalize,
    index = test_df.index)
join_df = test_df[test_df.columns.
    difference(cols_normalize)].join(norm_test_df)
test_df = join_df.reindex(columns = test_df.columns)
test_df = test_df.reset_index(drop=True)

#%%
rul = pd.DataFrame(test_df.groupby('id')['cycle'].max()).reset_index()
rul.columns = ['id', 'max']
truth_df.columns = ['more']
truth_df['id'] = truth_df.index + 1
truth_df['max'] = rul['max'] + truth_df['more']
truth_df.drop('more', axis = 1, inplace=True)

test_df = test_df.merge(truth_df,
    on=['id'], how='left')
test_df['RUL'] = test_df['max'] - test_df['cycle']
test_df.drop('max',
    axis = 1,
    inplace = True)

test_df['label1'] = np.where(test_df['RUL'] <= w1, 1, 0)
test_df.head()



#%% [markdown]
# ## LSTM time window
# window size = 50

#%%
def gen_sequence(id_df, seq_length, seq_cols):
    data_matrix = id_df[seq_cols].values
    num_elements = data_matrix.shape[0]

    for start, stop in zip(range(0, num_elements - seq_length),
        range(seq_length, num_elements)):
        yield data_matrix[start:stop, :]



#%%
def gen_labels(id_df, seq_length, label):
    data_matrix = id_df[label].values
    num_elements = data_matrix.shape[0]
    return data_matrix[seq_length:num_elements, :]

#%% [markdown]
# ## Generating the training sequence and corresponding label for the data

#%%
sequence_length = 50

sensor_cols = ['s' + str(i) for i in range(1, 22)]
sequence_cols = ['setting1', 'setting2', 'setting3', 'cycle_norm']
sequence_cols.extend(sensor_cols)
#%%
seq_gen = (list(gen_sequence
        (train_df[train_df['id']==id],
        sequence_length, sequence_cols))
        for id in train_df['id'].unique())

#%%
seq_array = np.concatenate(list(seq_gen)).astype(np.float32)
print(seq_array.shape)

#%%
label_gen = [gen_labels(train_df[train_df['id']==id],
        sequence_length, ['label1'])
        for id in train_df['id'].unique()]
label_array = np.concatenate(label_gen).astype(np.float32)
print(label_array.shape)

#%%
