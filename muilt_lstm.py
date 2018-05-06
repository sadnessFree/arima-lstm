from math import sqrt
import math
from numpy import concatenate
from matplotlib import pyplot
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import numpy


def mean_a_p_e(y_true, y_pred):
    sum = 0
    for i in range(len(y_true)):
        sum += math.fabs(y_pred[i] - y_true[i]) / y_true[i]
    return sum / len(y_true)


# convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg


# # load dataset
# dataset = read_csv('/Users/daihanru/Desktop/arima-lstm/DataSet/FEB15-2.csv', usecols=[2, 3], header=0,
#                    index_col=0)
# values = dataset.values
# values = values.astype('float32')
# # normalize features
# scaler = MinMaxScaler(feature_range=(0, 1))
# scaled = scaler.fit_transform(values)
# # frame as supervised learning
# reframed = series_to_supervised(scaled, 10, 1)
# # drop columns we don't want to predict
# # reframed.drop(reframed.columns[[4, 5]], axis=1, inplace=True)
# # reframed.drop(reframed.columns[[3, 4, 5, 6, 7, 8, 10, 11]], axis=1, inplace=True)
# print(reframed.head())
#
# # split into train and test sets
# values = reframed.values
# train_start = 0
# train_size = 1000
# train = values[train_start:train_start + train_size, :]
# test = values[1300:1400, :]
# # split into input and outputs
# train_X, train_y = train[:, :-1], train[:, -1]
# test_X, test_y = test[:, :-1], test[:, -1]
# # reshape input to be 3D [samples, timesteps, features]
# train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
# test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
# print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)
#
# # design network
# model = Sequential()
# model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
# model.add(Dense(1))
# model.compile(loss='mean_squared_error', optimizer='adam')
# # fit network
# history = model.fit(train_X, train_y, epochs=500, batch_size=20, validation_data=(test_X, test_y), verbose=2,
#                     shuffle=False)
# model.save('../model/lstm.h5')
# # plot history
# pyplot.plot(history.history['loss'], label='train')
# pyplot.plot(history.history['val_loss'], label='test')
# pyplot.legend()
# pyplot.show()
#
# # make a prediction
# yhat = model.predict(test_X)
# test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))
# # invert scaling for forecast
# inv_yhat = concatenate((yhat, test_X[:, 1:]), axis=1)
# inv_yhat = scaler.inverse_transform(inv_yhat)
# inv_yhat = inv_yhat[:, 0]
# # invert scaling for actual
# test_y = test_y.reshape((len(test_y), 1))
# inv_y = concatenate((test_y, test_X[:, 1:]), axis=1)
# inv_y = scaler.inverse_transform(inv_y)
# inv_y = inv_y[:, 0]
# # calculate RMSE
# mse = mean_squared_error(inv_y, inv_yhat)
# rmse = math.sqrt(mean_squared_error(inv_y, inv_yhat))
# mae = mean_absolute_error(inv_y, inv_yhat)
# mape = mean_a_p_e(inv_y, inv_yhat)
# print('Test MAE:%.3f MSE: %.3f RMSE:%.3f MAPE:%.3f' % (mae, mse, rmse, mape))
# pyplot.plot(inv_y)
# pyplot.plot(inv_yhat)
# pyplot.show(figsize=(12, 6))
