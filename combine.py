import pandas
import numpy
import math
import matplotlib.pyplot as plt
import arima
from keras.models import load_model
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from muilt_lstm import series_to_supervised

look_back = 10

# 加载数据
dataframe = pandas.read_csv('/Users/daihanru/Desktop/arima-lstm/DataSet/FEB15.csv', usecols=[2, 3], engine='python',
                            header=0, index_col=0)
values = dataframe.values
values = values.astype('float32')
# normalize features
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)
# frame as supervised learning
reframed = series_to_supervised(scaled, look_back, 1)
# drop columns we don't want to predict
# reframed.drop(reframed.columns[[4, 5]], axis=1, inplace=True)
# reframed.drop(reframed.columns[[3, 4, 5, 6, 7, 8, 10, 11]], axis=1, inplace=True)
print(reframed.head())

values = reframed.values
lstm_size = arima.size - look_back
test = values[lstm_size:lstm_size + arima.test_size + 1, :]
test_X, test_y = test[:, :-1], test[:, -1]
# reshape input to be 3D [samples, timesteps, features]
test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
print(test_X.shape, test_y.shape)

lstm_model = load_model('../model/lstm.h5')

# make a prediction
yhat = lstm_model.predict(test_X)
test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))
# invert scaling for forecast
inv_yhat = numpy.concatenate((yhat, test_X[:, 1:]), axis=1)
inv_yhat = scaler.inverse_transform(inv_yhat)
inv_yhat = inv_yhat[:, 0]
# invert scaling for actual
test_y = test_y.reshape((len(test_y), 1))
inv_y = numpy.concatenate((test_y, test_X[:, 1:]), axis=1)
inv_y = scaler.inverse_transform(inv_y)
inv_y = inv_y[:, 0]
# calculate RMSE
rmse = math.sqrt(mean_squared_error(inv_y, inv_yhat))
print('LSTM Test RMSE: %.3f' % rmse)

combined = list()
for i in range(len(inv_yhat)):
    combined.append((inv_yhat[i] + arima.predictions[i]) / 2)
rmse = math.sqrt(mean_squared_error(inv_y, combined))
print('EM combined Test RMSE: %.3f' % rmse)

plt.plot(inv_y, '-', label="real flow")
plt.plot(arima.predictions, '--', color='y', label="ARIMA")
plt.plot(inv_yhat, '--', color='red', label="LSTM")
plt.plot(combined, 'x-', color='g', label="combined")
plt.legend(loc='upper left')
plt.show()
