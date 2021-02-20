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
import bs4 as bs
import pickle
import requests
import datetime as dt
import os
import pandas as pd
import pandas_datareader.data as web
from datetime import date


import bs4 as bs
import pickle
import requests

def save_tickers():
    resp=requests.get('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
    soup=bs.BeautifulSoup(resp.text)
    table=soup.find('table',{'class':'wikitable sortable'})
    tickers=[]

    for row in table.findAll('tr')[1:]:
        ticker=row.findAll('td')[0].text[:-1]
        tickers.append(ticker)

    files = ['TRANS', 'XLF', 'SMH', 'SPY', 'IWM' ]
    for x in files:
        tickers.append(x)
    with open("tickers.pickle",'wb') as f:
        pickle.dump(tickers, f)
        return tickers

tickers=[]
tickers = save_tickers()

print(tickers)
today = date.today()

def fetch_data():
    with open("tickers.pickle",'rb') as f:
        tickers=pickle.load(f)

if not os.path.exists('stock_details'):
    os.makedirs('stock_details')
    count=200
start= dt.datetime(2017,1,1)
end=today
count=0

if os.path.getctime('stock_details') == dt.datetime.strptime(today.ctime(), "%a %b %d %H:%M:%S %Y"):
    for ticker in tickers:
        count+=1
        print(ticker)

        try:
            df=web.DataReader(ticker, 'yahoo', start, end)
            df.to_csv('stock_details/{}.csv'.format(ticker))
        except:
            print("Error")
            continue

fetch_data()
#       transportation     Banking      Semiconductor

# for f in files:
#     temp = pd.read_csv(f)
#     temp = temp.drop('Date', axis=1)
#     temp = temp.drop(' Volume', axis=1) #put back eventually
#
#         # print(y_min, coeff)
#         # raise Exception("stop")
#     normalized_temp=(temp-temp.min())/(temp.max()-temp.min())
#     temp = normalized_temp
#     temp['MA'] = temp[' Close/Last'].values[ ::-1]
#     temp['STD'] = temp[' Close/Last'].values[ ::-1]
#     temp['MA'] = temp['MA'].rolling(window=10).mean()
#     temp['STD'] = temp['STD'].rolling(window=10).std()
#     temp['MA'] = temp['MA'].values[ ::-1]
#     temp['STD'] = temp['STD'].values[ ::-1]
#     df = df.join(temp, rsuffix='_'+f)
df = None
for ticker in tickers:
    if not os.path.exists('../Video #1/stock_details/{}.csv'.format(ticker)):
        continue
    temp = pd.read_csv('../Video #1/stock_details/{}.csv'.format(ticker))

    temp = temp.drop('Date', axis=1)
    temp = temp.drop('Volume', axis=1) #put back eventually
    temp = temp.drop('Adj Close', axis=1)

    if ticker == 'SPY':
        y_min = temp['Close'].min()
        coeff = temp['Close'].max() - temp['Close'].min()
    #NORMALIZE male nail polish
    normalized_temp=(temp-temp.min())/(temp.max()-temp.min())
    temp = normalized_temp
    temp['MA'] = temp['Close'].values[ ::-1]
    temp['STD'] = temp['Close'].values[ ::-1]
    temp['MA'] = temp['MA'].rolling(window=10).mean()
    temp['STD'] = temp['STD'].rolling(window=10).std()
    temp['MA'] = temp['MA'].values[ ::-1]
    temp['STD'] = temp['STD'].values[ ::-1]
    if df is not None:
        df = df.join(temp, rsuffix='_'+ticker+'.csv')
    else:
        df = temp

#2. Do classifier dataset(SPY)
df['spy'] =  df['Close_SPY.csv']

df['spy'] = df['spy'].shift(1)  #in order to train we need to offset by 1 so it can "see" the next day's closing / opening in future implem.
df = df.dropna() #drop any NaN values

X = df.drop('spy',axis=1)
Y = df['spy'].shift(1)
# y_min = Y.min()
# coeff = Y.max() - Y.min()
# normalized_temp=(Y-y_min)/coeff  # ( Y.max() - Y.min()) + Y.min()
# Y = normalized_temp
# print(X)
# print(Y)
#timeInput, batch, epoch
# timeInputs = [1, 2, 3, 4, 5, 6, 7, 8, 10, 20, 25]
# batchs = [1, 5, 10, 50, 100, 500, 1000]
# epochs = [ 1, 5, 10, 20, 50, 100, 500, 1000]


timeInputs = [5,7 ]
batchs = [1000]
epochs = [500]
max_R2S = -1
max_XVS = -1
best = [-1,-1,-1]
boo = []
Y_train = []
x_vec = X.to_numpy()
y_vec = Y.to_numpy()



for timeNum in timeInputs:
    boo.clear()
    Y_train.clear()

    for i in range(timeNum,len(X)):
        boo.append(x_vec[i-timeNum:i,:])
        Y_train.append(y_vec[i])
    b, y = np.array(boo), np.array(Y_train)
    ################### ML MODEL CREATION ############################################
    X_train, X_test, y_train, y_test = train_test_split(b, y, test_size=0.2)
    del b, y

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
        mod.add(LSTM(units = 64, dropout = 0.3, return_sequences=True, input_shape = (timeNum, 3042)))
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

    for epochNum in epochs:
        for batchNum in batchs:
            # regressor = KerasRegressor(build_fn=ann_model, batch_size=4,epochs=500)
            # callback=tf.keras.callbacks.ModelCheckpoint(filepath='regression_model',
            #                                            monitor='mean_absolute_error',
            #                                            verbose=1,
            #                                            save_best_only=True,
            #                                            save_weights_only=False,
            #                                            mode='auto')
            # results=regressor.fit(X_train,y_train,callbacks=[callback])

            RNN_model = model()
            callback=tf.keras.callbacks.ModelCheckpoint(filepath='./RNN_model.h5',
                                                       monitor= 'mean_squared_error',
                                                       verbose=1,
                                                       save_best_only=True,
                                                       save_weights_only=False,
                                                       mode='auto',
                                                       save_freq='epoch')
            RNN_model.fit(X_train, y_train, epochs = epochNum , batch_size = batchNum,callbacks=[callback])
            #RNN_model = tf.keras.models.load_model('./RNN_model.h5')

            #############################################
            #  #               testing             #   #
            #############################################

            y_pred = []
            y_valu = []
            y_pred = RNN_model.predict(X_test)
            y_pred = (y_pred * coeff) + y_min #reverse the normalization

            y_valu = y_test
            y_valu = (y_valu * coeff) + y_min

            print(y_pred)
            print(y_valu)

            XVS = explained_variance_score(y_valu, y_pred)
            R2S = r2_score(y_valu, y_pred)

            if max_R2S < R2S:
                max_R2S = R2S
                max_XVS = XVS
                best = [timeNum, epochNum, batchNum]
                print("ran BEST: ")
                print(best)
                print(max_R2S, max_XVS)

            print(best)
            print(max_R2S, max_XVS)

            print(" _______________________________________")
            print(" ____________  Parameters : t, E, B ____")
            print(" _______________________________________")

            print(timeNum, epochNum, batchNum)


    # plotting the line 1 points

    print(best)
    print(max_R2S, max_XVS)
    # plt.plot(np.arange(0, 45), y_spy, label = "prediction")
    #
    # # plotting the line 2 points
    # plt.plot(np.arange(0, 45), y_test, label = "test")
    # plt.xlim(0,100)
    # # naming the x axis
    # plt.xlabel('x - axis')
    # # naming the y axis
    # plt.ylabel('y - axis')
    # # giving a title to my graph
    # plt.title('Two lines on same graph!')
    #
    # # show a legend on the plot
    # plt.legend()

    # function to show the plot
    #plt.show()
