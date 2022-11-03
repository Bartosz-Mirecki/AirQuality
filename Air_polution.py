import os
import datetime
from pickle import TRUE


import IPython
import IPython.display
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import pickle 
from tensorflow import keras
import platform

mpl.rcParams['figure.figsize'] = (8, 6)
mpl.rcParams['axes.grid'] = False

def prep_data(data_set):
    #Delete the few last rows which contain NaN values
    data_set.dropna(subset = ['CO(GT)'], inplace=True)

    #Delete the specyfic columns# Some of them contains NaN values, and another are 
    #problematic for parameters machine learning, because contains 0
    data_set.drop(['Unnamed: 15','Unnamed: 16','NMHC(GT)','NO2(GT)'],axis=1, inplace=True)

    #Preparing date and time columns to generate timestamp
    data_set.insert(loc = 0, column='Date_time', value=data_set['Date'].astype(str) +" "+ data_set['Time'])

    data_set.drop(['Date','Time'],axis=1, inplace=True)

    date = pd.to_datetime(data_set.pop('Date_time'), format='%d/%m/%Y %H.%M.%S')

    data_set.insert(loc = 0, column='Date_time', value = date)

    
    plot_cols = ['PT08.S2(NMHC)','NOx(GT)','PT08.S1(CO)']
    plot_features = data_set[plot_cols]
    plot_features.index = date
    plot_features.plot(subplots=True)

    plot_features = data_set[plot_cols][:480]
    plot_features.index = date[:480]
    plot_features.plot(subplots=True)

    plt.show()

    print('transpose data set before cope with invalid values:')
    print(data_set.describe().transpose())

    #change the -200 values to 0, for better suitable parameter
    #every column has the min value -200

    for col in data_set:
        op = data_set[col]
        bad_op = op == -200
        good_op = op != -200
        op[bad_op] = op[good_op].mean()

    print('transpose data set after cope with invalid values:')
    print(data_set.describe().transpose())

    timestamp_s = data_set['Date_time'].map(pd.Timestamp.timestamp)

    day = 24*60*60
    year = (365.2425)*day

    data_set['Day sin'] = np.sin(timestamp_s * (2 * np.pi / day))
    data_set['Day cos'] = np.cos(timestamp_s * (2 * np.pi / day))
    data_set['Year sin'] = np.sin(timestamp_s * (2 * np.pi / year))
    data_set['Year cos'] = np.cos(timestamp_s * (2 * np.pi / year))

    plot_cols = ['Day sin','Day cos','Year sin','Year cos']
    plot_features = data_set[plot_cols]
    plot_features.index = date
    plot_features.plot(subplots=True)

    plt.show()

    if platform.system()=='Darwin' or platform.system()=='Linux':
      data_set.to_csv(os.path.dirname(__file__)+'/AirQualityUCI/AirQualityUCI(cleared).csv',index=False)
    elif platform.system()=='Windows':
      data_set.to_csv(os.path.dirname(__file__)+'\AirQualityUCI\AirQualityUCI(cleared).csv',index=False)

    data_set.drop(['Date_time'],axis=1, inplace=True)
    return data_set
    

def deviding_data_set(data_set):

    #preparing data for training(Spliting data: 70:20:10, training, validation, test)
    train_df = data_set[0:int(0.7*len(data_set))]
    val_df = data_set[int(0.7*len(data_set)):int(0.9*len(data_set))]
    test_df = data_set[int(0.9*len(data_set)):len(data_set)]

    column_indices = {name: i for i, name in enumerate(data_set.columns)}

    print(column_indices)

    train_mean = train_df.mean()
    train_std = train_df.std()

    train_df = (train_df-train_mean)/train_std
    val_df = (val_df-train_mean)/train_std
    test_df = (test_df-train_mean)/train_std

    ax = sns.violinplot(data=train_df[['CO(GT)', 'PT08.S1(CO)', 'C6H6(GT)',
        'PT08.S2(NMHC)', 'NOx(GT)', 'PT08.S3(NOx)', 'PT08.S4(NO2)',
        'PT08.S5(O3)', 'T', 'RH', 'AH','Day sin', 'Day cos', 'Year sin', 'Year cos']])

    plt.show()

    return train_df, val_df, test_df

#Load data from csv file with specific separator ';'

if platform.system()=='Darwin' or platform.system()=='Linux':
  data_set = pd.read_csv(os.path.dirname(__file__)+'/AirQualityUCI/AirQualityUCI.csv',sep=';',decimal = ',')  
elif platform.system()=='Windows':
  data_set = pd.read_csv(os.path.dirname(__file__)+'\AirQualityUCI\AirQualityUCI.csv',sep=';',decimal = ',')  

data_set = prep_data(data_set)

train_df, val_df, test_df = deviding_data_set(data_set)

class WindowGenerator():
    def __init__(self, label_width, shift, input_width,
                 train_df = train_df, val_df = val_df, 
                 test_df = test_df, label_columns = None):
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df

        self.label_columns = label_columns
        if label_columns is not None:
            self.label_columns_indices = {name: i for i, name in
                                        enumerate(label_columns)}
        self.column_indices = {name: i for i, name in
                           enumerate(train_df.columns)}

        #Parameters
    
        self.input_width = input_width
        self.shift = shift
        self.label_width = label_width
        self.total_width = input_width + shift

        self.input_slice = slice(0,self.input_width)
        self.input_indices = np.arange(self.input_width)[self.input_slice]


        self.labels_slice = slice(self.total_width - self.label_width, self.total_width)
        self.label_indices = np.arange(self.total_width)[self.labels_slice]
        

    def __repr__(self):
        return '\n'.join([
            f'Input width: {self.input_width}',
            f'Shift: {self.shift}',
            f'Label width: {self.label_width}',
            f'Total width: {self.total_width}',
            f'Input_slice: {self.input_slice}',
            f'Input_indices: {self.input_indices}',
            f'Label_slice: {self.labels_slice}',
            f'Label_indices: {self.label_indices}',
        ])
    


# WINDOW GENERATOR #################################################################################
OUT_STEPS = 24
w = WindowGenerator(label_width=OUT_STEPS, shift=OUT_STEPS, input_width=24,label_columns=['PT08.S1(CO)'])

print(w)

def plot(self, model=None, plot_col='PT08.S1(CO)', max_subplots=5):
  inputs, labels = self.example
  plt.figure(figsize=(12, 8))
  plot_col_index = self.column_indices[plot_col]
  max_n = min(max_subplots, len(inputs))
  for n in range(max_n):
    plt.subplot(max_n, 1, n+1)
    plt.ylabel(f'{plot_col} [normed]')
    plt.plot(self.input_indices, inputs[n, :, plot_col_index],
             label='Inputs', marker='.', zorder=-10)

    if self.label_columns:
      label_col_index = self.label_columns_indices.get(plot_col, None)
    else:
      label_col_index = plot_col_index

    if label_col_index is None:
      continue

    plt.scatter(self.label_indices, labels[n, :, label_col_index],
                marker='P', edgecolors='black', label='Labels', c='dodgerblue', s=64)
    if model is not None:
      predictions = model(inputs)
      plt.scatter(self.label_indices, predictions[n, :, label_col_index],
                  marker='o', edgecolors='black', label='Predictions',
                  c='gold', s=64)

    if n == 0:
      plt.legend()

  plt.xlabel('Time [h]')

WindowGenerator.plot = plot

def split_window(self, features):
  inputs = features[:, self.input_slice, :]
  labels = features[:, self.labels_slice, :]
  if self.label_columns is not None:
    labels = tf.stack(
        [labels[:, :, self.column_indices[name]] for name in self.label_columns],
        axis=-1)

  # Slicing doesn't preserve static shape information, so set the shapes
  # manually. This way the `tf.data.Datasets` are easier to inspect.
  inputs.set_shape([None, self.input_width, None])
  labels.set_shape([None, self.label_width, None])

  return inputs, labels

WindowGenerator.split_window = split_window

def make_dataset(self, data):
    data = np.array(data, dtype=np.float32)
    ds = tf.keras.utils.timeseries_dataset_from_array(
        data=data,
        targets=None,
        sequence_length=self.total_width,
        sequence_stride=1,
        shuffle=True,
        batch_size=4,)

    ds = ds.map(self.split_window)

    return ds

WindowGenerator.make_dataset = make_dataset

@property
def train(self):
  return self.make_dataset(self.train_df)

@property
def val(self):
  return self.make_dataset(self.val_df)

@property
def test(self):
  return self.make_dataset(self.test_df)

@property
def example(self):
  result = getattr(self, '_example', None)
  if result is None:
    # No example batch was found, so get one from the `.train` dataset
    result = next(iter(self.train))
    # And cache it for next time
    self._example = result
  return result



WindowGenerator.train = train
WindowGenerator.val = val
WindowGenerator.test = test
WindowGenerator.example = example

print(f'this is the w.exaple{w.example}')
num_features = data_set.shape[1]

CONV_WIDTH = 3
CNN = tf.keras.Sequential([
    # Shape [batch, time, features] => [batch, CONV_WIDTH, features]
    tf.keras.layers.Lambda(lambda x: x[:, -3:, :]),
    # Shape => [batch, 1, conv_units]
    tf.keras.layers.Conv1D(256, activation='relu', kernel_size=(3)),
    # Shape => [batch, 1,  out_steps*features]
    tf.keras.layers.Dense(OUT_STEPS*num_features,
                          kernel_initializer=tf.initializers.zeros()),
    # Shape => [batch, out_steps, features]
    tf.keras.layers.Reshape([OUT_STEPS, num_features])

])

print('Input shape:', w.example[0].shape)
print('Output shape:', CNN(w.example[0]).shape)

MAX_EPOCHS = 100

print(CNN.summary())


def compile_and_fit(model, window, patience=15):
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                        patience=patience,
                                                        mode='min')

    model.compile(loss=tf.keras.losses.MeanSquaredError(),
                    optimizer=tf.keras.optimizers.Adam(),
                    metrics=[tf.keras.metrics.MeanAbsoluteError()])

    history = model.fit(window.train, epochs=MAX_EPOCHS,
                        validation_data=window.val,
                        callbacks=[early_stopping])
    return history

history = compile_and_fit(CNN, w)

if platform.system()=='Darwin' or platform.system()=='Linux':
  CNN.save(os.path.dirname(__file__)+'/my_model.h5')
elif platform.system()=='Windows':
  CNN.save(os.path.dirname(__file__)+'\my_model.h5')



IPython.display.clear_output()

val_performance = {}
performance = {}
val_performance['Conv'] = CNN.evaluate(w.val)
performance['Conv'] = CNN.evaluate(w.test, verbose=0)

plt.plot(history.history['loss'], label = 'loss')
plt.plot(history.history['val_loss'], label = 'val_loss')
plt.legend()

plt.show()

w.plot(CNN)

plt.show()

input_ex,label_ex = w.example

print(input_ex.shape)

print(CNN(input_ex).shape)


print(CNN(next(iter(w.train))[0])[0])

loss = history.history['loss']


if platform.system()=='Darwin' or platform.system()=='Linux':
  with open(os.path.dirname(__file__)+'/CNN_history', 'wb') as fout:
    pickle.dump(loss, fout)
elif platform.system()=='Windows':
  with open(os.path.dirname(__file__)+'\CNN_history', 'wb') as fout:
    pickle.dump(loss, fout)
