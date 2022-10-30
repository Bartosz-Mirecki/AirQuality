import os
import datetime
import IPython
import IPython.display
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from datetime import datetime
import json
import pickle
import platform

if platform.system()=='Darwin' or platform.system()=='Linux':
  df = pd.read_csv(os.path.dirname(__file__)+'/AirQualityUCI/AirQualityUCI(cleared).csv')  
elif platform.system()=='Windows':
  df = pd.read_csv(os.path.dirname(__file__)+'\AirQualityUCI\AirQualityUCI(cleared).csv')  

new_model = tf.keras.models.load_model('my_model.h5')
new_model.summary()

df.insert(loc = 0, column = 'Date', value = pd.to_datetime(df.pop('Date_time'), format='%Y-%m-%d %H:%M:%S'))

date_time = df.pop('Date')
mean = df.mean()
std = df.std()
df = (df - mean)/std
df = tf.expand_dims(df, 0)

print(std)
print(mean)


midnight_loc = date_time.dt.hour == 0

my_table=[]


for i in range(midnight_loc[midnight_loc==True].index[-1]-1):

    if(midnight_loc[i] == True):my_table.append({
            "Date": str("{:02d}".format(date_time._get_value(i,'Date').day))+"-"+str("{:02d}".format(date_time._get_value(i,'Date').month))+"-"+str(date_time._get_value(i,'Date').year),
            "Input": (df[:,i:i+24,1]*std[1]+mean[1]).numpy(),
            "Output": ((new_model(df[:,i:i+24,:])[:,:,1])*std[1]+mean[1]).numpy(),
            "Labels": (df[:,i+24:i+48,1]*std[1]+mean[1]).numpy()
        })

print(type(my_table[0]["Input"]))

plt.plot(my_table[0]["Labels"][0,:])

plt.show()


if platform.system()=='Darwin' or platform.system()=='Linux':
  with open(os.path.dirname(__file__)+'/outfile', 'wb') as fout:
    pickle.dump(my_table, fout)
elif platform.system()=='Windows':
  with open(os.path.dirname(__file__)+'\outfile', 'wb') as fout:
    pickle.dump(my_table, fout)
