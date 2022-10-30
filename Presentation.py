import os
import datetime
import IPython
import IPython.display
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import pickle
import streamlit as st
import platform

st.title('Prediction of the air pollution parameter')
st.subheader('PT08.S1 (tin oxide)')

'Check the cleared data'

if platform.system()=='Darwin' or platform.system()=='Linux':
    df = pd.read_csv(os.path.dirname(__file__)+'/AirQualityUCI/AirQualityUCI(cleared).csv')
    f = open(os.path.dirname(__file__)+'/outfile', 'rb')
    final_data = pickle.load(f) 
elif platform.system()=='Windows':
    df = pd.read_csv(os.path.dirname(__file__)+'/AirQualityUCI/AirQualityUCI(cleared).csv')
    f = open(os.path.dirname(__file__)+'/outfile', 'rb')
    final_data = pickle.load(f)  



col_names = df.keys()

parameter = st.selectbox(
    'Select the parameter',
    (col_names[1:12]))

time_range = st.selectbox(
    'Select time range',
    ('week','day'))

date = st.text_input('Type a date (format dd-mm-yyyy) range: 11-03-2004 to 03-04-2005', '11-03-2004')

df.insert(loc = 0, column = 'Date', value = pd.to_datetime(df.pop('Date_time'), format='%Y-%m-%d %H:%M:%S'))

cos = []

for i in range(len(df)):
    cos.append(str("{:02d}".format(df._get_value(i,'Date').day))+"-"+str("{:02d}".format(df._get_value(i,'Date').month))+"-"+str("{:02d}".format(df._get_value(i,'Date').year)))
    
df.insert(loc = 0, column = 'Date_time', value = cos)

start_ind = (df.index[df['Date_time'] == date])[0]

if time_range == 'week':
    y_values = df[start_ind:start_ind+24*7][parameter]
elif time_range =='day':
    y_values = df[start_ind:start_ind+24][parameter]

'\n'
'\n'

st.line_chart(y_values)

latext = r'''
    $$ 
    DataFrame = \frac{(DataFrame-DataFrame.mean)} {DataFrame.std}
    $$ 
    '''
st.subheader('Parameters normalization')

st.write(latext)

df_norm = df[col_names[1:12]]

df_norm = (df_norm-df_norm.mean())/df_norm.std()

'\n'
'\n'

fig = plt.figure(figsize=(10, 4))
sns.violinplot(data=df_norm[col_names[1:12]])
plt.xticks(rotation=45)

st.pyplot(fig)

'\n'
'\n'

st.subheader('CNN Architecture')

d = {'Layer (type)': ['lambda (Lambda)', 'conv1d (Conv1D)','dense (Dense)','reshape (Reshape)'],
     'Output Shape': ['(None, 3, 15)', '(None, 1, 256)','(None, 1, 360)','(None, 24, 15)'],
     'Parameters': ['0', '11776','92520','0']}

d = pd.DataFrame(data=d)
d

f = open('/Users/bartoszmirecki/Documents/Coding/time_series_forecasting_project/CNN_history', 'rb')
loss = pickle.load(f)

st.subheader('Learning loss plot')

history_loss = pd.DataFrame({'loss':loss})

st.line_chart(history_loss)
f.close()


python_indices  = [index for (index, item) in enumerate(final_data) if item["Date"] == date][0]

print((python_indices))

input_index = list(range(1, 25))
label_index = list(range(25, 49))
prediction_index = list(range(25, 49))

st.subheader('Prediction')

fig = plt.figure(figsize=(10, 2))
plt.scatter(input_index,final_data[python_indices]["Input"],
                marker='P', edgecolors='black', label='Labels', c='dodgerblue', s=64)

plt.scatter(label_index,final_data[python_indices]["Labels"],
                marker='o', edgecolors='black', label='Predictions',
                  c='gold', s=64)

plt.scatter(prediction_index,final_data[python_indices]["Output"],
                marker='o', edgecolors='black', label='Labels', c='dodgerblue', s=64)

plt.legend()

st.pyplot(fig)



