#This is my Python Project -- start 12/12/2020


# best ways to code/learn --> #1 follow documentations and build your own structure and stuff                                           (best for full-understanding and scaling)
#                             #2 go along a video or website and thouroughly investigate their methods along with documentation         (what I am doing ) == best for speed and learning
#                             #3 go along a video build with them and change parameters/code to explore effects                         (good for analyzing results and broad concepts)
#                             #4 cheat. You get very little out of it but you can still learn about concepts                            (not good lol)

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten, Dropout, LSTM
from tensorflow.keras.layers import LayerNormalization
from tensorflow.keras.metrics import RootMeanSquaredError
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, explained_variance_score, r2_score
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import matplotlib.pyplot as plt

#       transportation     Banking      Semiconductor
files = ['TRANS_HIST.csv', 'XLF_HIST.csv', 'SMH_HIST.csv' ]
df  = pd.read_csv('IWM_HIST.csv') #creates the dataset with mid_cap data
df = df.drop('Date', axis=1)
df = df.drop(' Volume', axis=1)
normalized_temp=(df-df.min())/(df.max()-df.min())
df = normalized_temp
df = df.add_suffix('_IWM_HIST.csv') #Keeps data consistent
df['MA_IWM_HIST.csv'] = df[' Close/Last_IWM_HIST.csv'].values[ ::-1]
df['STD_IWM_HIST.csv'] = df[' Close/Last_IWM_HIST.csv'].values[ ::-1]
df['MA_IWM_HIST.csv'] = df['MA_IWM_HIST.csv'].rolling(window=20).mean()
df['STD_IWM_HIST.csv'] = df['STD_IWM_HIST.csv'].rolling(window=20).std()
df['MA_IWM_HIST.csv'] = df['MA_IWM_HIST.csv'].values[ ::-1]
df['STD_IWM_HIST.csv'] = df['STD_IWM_HIST.csv'].values[ ::-1]

for f in files:
    temp = pd.read_csv(f)
    temp = temp.drop('Date', axis=1)
    temp = temp.drop(' Volume', axis=1) #put back eventually
    normalized_temp=(temp-temp.min())/(temp.max()-temp.min())
    temp = normalized_temp
    temp['MA'] = temp[' Close/Last'].values[ ::-1]
    temp['STD'] = temp[' Close/Last'].values[ ::-1]
    temp['MA'] = temp['MA'].rolling(window=20).mean()
    temp['STD'] = temp['STD'].rolling(window=20).std()
    temp['MA'] = temp['MA'].values[ ::-1]
    temp['STD'] = temp['STD'].values[ ::-1]
    df = df.join(temp, rsuffix='_'+f)

#2. Do classifier dataset(SPY)
temp = pd.read_csv('SPY_HIST.csv')
df['spy'] =  temp[' Close/Last']

df['spy'] = df['spy'].shift(1)  #in order to train we need to offset by 1 so it can "see" the next day's closing / opening in future implem.
df = df.dropna() #drop any NaN values
X = df.drop('spy',axis=1)
Y = df['spy']
y_min = Y.min()
coeff = Y.max() - Y.min()
normalized_temp=(Y-y_min)/coeff  #// * ( Y.max() - Y.min()) + Y.min()
Y = normalized_temp

boo = []
Y_train = []
x_vec = X.to_numpy()
y_vec = Y.to_numpy()
for i in range(10,len(X)):
    boo.append(x_vec[i-10:i,:])
    Y_train.append(y_vec[i])
b, y = np.array(boo), np.array(Y_train)
################### ML MODEL CREATION ############################################
X_train, X_test, y_train, y_test = train_test_split(b, y, test_size=0.2)

#sets the data to 3d for LSTM model
# boo = []
# Y_train = []
# x_vec = X_train.to_numpy()
# y_vec = y_train.to_numpy()
# for i in range(10,len(X_train)):
#     boo.append(x_vec[i-10:i,:])
#     Y_train.append(y_vec[i])
# b, y_train = np.array(boo), np.array(Y_train)
#b = b[:,np.newaxis, : ]   # [datarows, time_interval, features]
print(X_train.shape)
print(y_train.shape)




#Okay so now I have my data set in 3d witha time interval of 1 (day).

# #Create the LSTM RNN Model
def model():
    mod=Sequential()
    mod.add(LSTM(units = 64, dropout = 0.3, return_sequences=True, input_shape = (10, 24)))
    mod.add(LayerNormalization())
    mod.add(LSTM(units = 64))
    mod.add(LayerNormalization())
    mod.add((Dense(64, activation='tanh')))
    mod.add(Dropout(0.4))
    mod.add(Dense(1, activation='linear'))

    mod.compile(loss='mean_squared_error', optimizer='adam', metrics=[RootMeanSquaredError(),'mean_squared_error'])
    mod.summary()
    return mod

# #Create the ANN Model
def ann_model():
    mod=Sequential()
    mod.add(Dense(32, kernel_initializer='normal',input_dim = 24, activation='relu'))
    mod.add(Dense(64, kernel_initializer='normal',activation='relu'))
    mod.add(Dropout(0.4))
    mod.add(Dense(128, kernel_initializer='normal',activation='relu'))
    mod.add(Dropout(0.4))
    mod.add(Dense(256, kernel_initializer='normal',activation='relu'))
    mod.add(Dropout(0.4))
    mod.add(Dense(1, kernel_initializer='normal',activation='linear'))  # needed for it to quantify the number for output

    mod.compile(loss='mean_absolute_error', optimizer='adam', metrics=['accuracy','mean_absolute_error'])
    mod.summary()
    return mod


# regressor = KerasRegressor(build_fn=ann_model, batch_size=4,epochs=500)
# callback=tf.keras.callbacks.ModelCheckpoint(filepath='regression_model',
#                                            monitor='mean_absolute_error',
#                                            verbose=1,
#                                            save_best_only=True,
#                                            save_weights_only=False,
#                                            mode='auto')
# results=regressor.fit(X_train,y_train,callbacks=[callback])

# RNN_model = model()
# callback=tf.keras.callbacks.ModelCheckpoint(filepath='./RNN_model.h5',
#                                            monitor= 'mean_squared_error',
#                                            verbose=1,
#                                            save_best_only=True,
#                                            save_weights_only=False,
#                                            mode='auto',
#                                            save_freq='epoch')
# RNN_model.fit(X_train, y_train, epochs = 100, batch_size = 3,callbacks=[callback])

RNN_model = tf.keras.models.load_model('./RNN_model.h5')

#############################################
#  #               testing             #   #
#############################################

y_pred = []
y_pred = RNN_model.predict(X_test)
y_spy = (y_pred * coeff) + y_min
print(y_spy)
y_test = (y_test * coeff) + y_min
print(y_test)
print(y_test.shape)

print(explained_variance_score(y_test, y_spy))
print(r2_score(y_test, y_spy))



# plotting the line 1 points


plt.plot(np.arange(0, 227), y_spy, label = "prediction")

# plotting the line 2 points
plt.plot(np.arange(0, 227), y_test, label = "test")
plt.xlim(0,100)
# naming the x axis
plt.xlabel('x - axis')
# naming the y axis
plt.ylabel('y - axis')
# giving a title to my graph
plt.title('Two lines on same graph!')

# show a legend on the plot
plt.legend()

# function to show the plot
#plt.show()
