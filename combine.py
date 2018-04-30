import pandas
import numpy
import math
import matplotlib.pyplot as plt
import arima
from keras.models import load_model
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from muilt_lstm import series_to_supervised

look_back = 8

# 加载数据
dataframe = pandas.read_csv('/Users/daihanru/Desktop/arima-lstm/DataSet/FEB15-2.csv', usecols=[2, 3], engine='python',
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
test = values[lstm_size:lstm_size + arima.test_size, :]
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
mse = mean_squared_error(inv_y, inv_yhat)
rmse = math.sqrt(mean_squared_error(inv_y, inv_yhat))
mae = mean_absolute_error(inv_y, inv_yhat)
mape = arima.mean_a_p_e(inv_y, inv_yhat)
print('LSTM Test MAE:%.3f MSE: %.3f RMSE:%.3f MAPE:%.3f' % (mae, mse, rmse, mape))
plt.plot(inv_y, '-', label="real flow")
plt.plot(inv_yhat, 'x--', color='red', label="LSTM")
plt.legend(loc='upper left')
plt.xlabel("period")
plt.ylabel("volume")
plt.ylim(0, 800)
plt.show(figsize=(12, 6))

combined = list()
for i in range(len(inv_yhat)):
    combined.append((inv_yhat[i] + arima.predictions[i]) / 2)
mse = mean_squared_error(inv_y, combined)
rmse = math.sqrt(mean_squared_error(inv_y, combined))
mae = mean_absolute_error(inv_y, combined)
mape = arima.mean_a_p_e(inv_y, combined)
print('EW combined Test MAE:%.3f MSE: %.3f RMSE:%.3f MAPE:%.3f' % (mae, mse, rmse, mape))

windows_size = 1
lstm_error_list = []
arima_error_list = []
lstm_weight = 1
arima_weight = 1
dyn_combined = list()
for i in range(len(inv_yhat)):
    lstm_error_list.append(math.fabs(inv_yhat[i] - inv_y[i]))
    arima_error_list.append(math.fabs(arima.predictions[i] - inv_y[i]))
    if len(lstm_error_list) >= windows_size:
        lstm_error = numpy.average(lstm_error_list[-windows_size:])
        arima_error = numpy.average(arima_error_list[-windows_size:])
        lstm_weight = (1 - lstm_error / (lstm_error + arima_error)) * 2
        arima_weight = 2 - lstm_weight
    dyn_combined.append((lstm_weight * inv_yhat[i] + arima_weight * arima.predictions[i]) / 2)

mse = mean_squared_error(inv_y, dyn_combined)
rmse = math.sqrt(mean_squared_error(inv_y, dyn_combined))
mae = mean_absolute_error(inv_y, dyn_combined)
mape = arima.mean_a_p_e(inv_y, dyn_combined)
print('dyn combined Test MAE:%.3f MSE: %.3f RMSE:%.3f MAPE:%.3f' % (mae, mse, rmse, mape))

plt.figure()
plt.plot(inv_y, '-', label="real flow")
plt.plot(arima.predictions, 'x--', color='y', label="ARIMA")
plt.plot(inv_yhat, 'x--', color='red', label="LSTM")
plt.plot(dyn_combined, 'x--', color='g', label="combined")
plt.legend(loc='upper left')
plt.xlabel("period")
plt.ylabel("volume")
plt.ylim(0, 800)
plt.show(figsize=(12, 6))
