import warnings
warnings.filterwarnings("ignore", message="numpy.*")

import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import numpy
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import GRU
from tensorflow.keras.layers import Bidirectional

import csv
import re
import math
from sklearn.metrics import mean_squared_error
import time
import keras
from keras import callbacks
import statsmodels.api as sm

print(tf.__version__)

path = '43.csv'

df = pd.read_csv(path)
print(df.shape)

cols = []

with open(path, 'r') as file:
    reader = csv.reader(file, delimiter=';') 
    c = 0
    for row in reader:
        if c==0:
            for i in range(len(row)):
                #cols.append(row[i])
                s = row[i]
                s = re.sub(r'\s', '', s)
                cols.append(s)
            c= c+1
value = []

with open(path, 'r') as file:
    reader = csv.reader(file, delimiter=';')
    c = 0
    for row in reader:
        if c==0:
            for i in range(len(row)):
                c = c+1
                continue
        else:
            val =[]
            for i in range(len(row)):
                s = row[i]
                s = re.sub(r'\s', '', s)
                val.append(float(s))
            value.append(val)
            val = []

df = pd.DataFrame(value, columns = cols) 
df.head()

prediction = 'Memoryusage[KB]'
time_step = 100
epochs = 5
loss_function = 'mean_squared_error'
file1_name  = 'Memoryusage[KB]_mean_squared_error_loss_val_loss_idealmodel'
file2_name = 'Memoryusage[KB]_mean_squared_error_CDF_idealmodel'
file3_name = 'Memoryusage[KB]_mean_squared_error_TestSet_Split_idealmodel'

#https://keras.io/api/losses/regression_losses/

df1=df.reset_index()[prediction]
sns.set(rc={'figure.figsize':(11, 4)})
print('before normalization')
plt.plot(df1)
plt.show()
scaler=MinMaxScaler(feature_range=(0,1))
df1=scaler.fit_transform(np.array(df1).reshape(-1,1))

print('after normalization')
plt.plot(df1)
plt.show()


#split training and test size
training_size=int(len(df1)*0.75)
test_size=len(df1)-training_size
train_data,test_data=df1[0:training_size,:],df1[training_size:len(df1),:1]

def create_dataset(dataset, time_step=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-time_step-1):
		a = dataset[i:(i+time_step), 0] 
		dataX.append(a)
		dataY.append(dataset[i + time_step, 0])
	return numpy.array(dataX), numpy.array(dataY)
 

X_train, y_train = create_dataset(train_data, time_step)
X_test, ytest = create_dataset(test_data, time_step)

X_train =X_train.reshape(X_train.shape[0],X_train.shape[1] , 1)
X_test = X_test.reshape(X_test.shape[0],X_test.shape[1] , 1)

model=Sequential()
model.add(Conv1D(filters=64, kernel_size=5, strides=1, padding="causal",activation="relu", input_shape=[100, 1]))
model.add(Bidirectional(LSTM(64, return_sequences=True)))
model.add(LSTM(64,return_sequences=True))
model.add(LSTM(64,return_sequences=True))
model.add(GRU(64, return_sequences=True))
model.add(LSTM(50))
model.add(Dense(1))
model.compile(loss=loss_function,optimizer='adam')