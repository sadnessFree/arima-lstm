import pandas
import numpy
import math
import matplotlib.pyplot as plt
import matplotlib
import arima
from HM import hm_p
from svr import svr_test
from keras.models import load_model
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from muilt_lstm import series_to_supervised
from sklearn import linear_model

matplotlib.rcParams['font.family'] = 'SimHei'

look_back = 10
line_test_size = 300
windows_size = 7
test_start = line_test_size + arima.windows_size - 1

# 加载数据
dataframe = pandas.read_csv('/Users/daihanru/Desktop/arima-lstm/DataSet/FEB15-2.csv', usecols=[2, 3], engine='python',
                            header=0, index_col=0)
values = dataframe.values
values = values.astype('float32')
# 正则化
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)

# frame as supervised learning
reframed = series_to_supervised(scaled, look_back, 1)
# drop columns we don't want to predict
# reframed.drop(reframed.columns[[4, 5]], axis=1, inplace=True)
# reframed.drop(reframed.columns[[3, 4, 5, 6, 7, 8, 10, 11]], axis=1, inplace=True)
print(reframed.head())

values = reframed.values
lstm_size = arima.start + arima.size - look_back
test = values[lstm_size:lstm_size + arima.test_size, :]
test_X, test_y = test[:, :-1], test[:, -1]
# reshape input to be 3D [samples, timesteps, features]
test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
print(test_X.shape, test_y.shape)

lstm_model = load_model('../model/lstm-15min.h5')

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
mse = mean_squared_error(inv_y[test_start:], inv_yhat[test_start:])
rmse = math.sqrt(mean_squared_error(inv_y[test_start:], inv_yhat[test_start:]))
mae = mean_absolute_error(inv_y[test_start:], inv_yhat[test_start:])
mape = arima.mean_a_p_e(inv_y[test_start:], inv_yhat[test_start:])
print('LSTM Test MAE:%.3f MSE: %.3f RMSE:%.3f MAPE:%.3f' % (mae, mse, rmse, mape))
plt.plot(inv_y[test_start:], '-', label="real flow")
plt.plot(inv_yhat[test_start:], '--', color='red', label="LSTM")
plt.legend(loc='upper right')
plt.xlabel("period(15-minute intervals)")
plt.ylabel("volume(vehicle/period)")
plt.ylim(0, 800)
plt.show(figsize=(12, 6))

combined = list()
for i in range(len(inv_yhat)):
    combined.append((inv_yhat[i] + arima.predictions[i]) / 2)
mse = mean_squared_error(inv_y[test_start:], combined[test_start:])
rmse = math.sqrt(mean_squared_error(inv_y[test_start:], combined[test_start:]))
mae = mean_absolute_error(inv_y[test_start:], combined[test_start:])
mape = arima.mean_a_p_e(inv_y[test_start:], combined[test_start:])
print('EW combined Test MAE:%.3f MSE: %.3f RMSE:%.3f MAPE:%.3f' % (mae, mse, rmse, mape))

dyn_combined = list()
model = linear_model.LinearRegression(normalize=True)
line_train_x = []
line_train_y = []
for i in range(line_test_size - windows_size + 1):
    for j in range(windows_size):
        line_train_x.append(arima.predictions[i + j][0])
    for j in range(windows_size):
        line_train_x.append(inv_yhat[i + j])
    line_train_y.append(inv_y[i + windows_size - 1])
line_train_x = numpy.array(line_train_x).reshape(line_test_size - windows_size + 1, 2 * windows_size)
model.fit(line_train_x, line_train_y)

line_test_x = []
for i in range(arima.test_size - line_test_size - windows_size + 1):
    for j in range(windows_size):
        line_test_x.append(arima.predictions[line_test_size + i + j][0])
    for j in range(windows_size):
        line_test_x.append(inv_yhat[line_test_size + i + j])
line_test_x = numpy.array(line_test_x).reshape(arima.test_size - line_test_size - windows_size + 1,
                                               2 * windows_size)
dyn_combined = model.predict(line_test_x)

mse = mean_squared_error(inv_y[test_start:], dyn_combined[arima.windows_size - windows_size:])
rmse = math.sqrt(mean_squared_error(inv_y[test_start:], dyn_combined[arima.windows_size - windows_size:]))
mae = mean_absolute_error(inv_y[test_start:], dyn_combined[arima.windows_size - windows_size:])
mape = arima.mean_a_p_e(inv_y[test_start:], dyn_combined[arima.windows_size - windows_size:])
print('dyn combined Test MAE:%.3f MSE: %.3f RMSE:%.3f MAPE:%.3f' % (mae, mse, rmse, mape))

plt.figure(figsize=(12, 6))
plt.plot(inv_y[test_start:], '-', label="real flow")
# plt.plot(arima.predictions, 'x--', color='y', label="ARIMA")
# plt.plot(inv_yhat, 'x--', color='red', label="LSTM")
plt.plot(dyn_combined[arima.windows_size - windows_size:], '--', color='red', label="arima-lstm-linear-7")
plt.legend(loc='upper right')
plt.xlabel("period(15-minute intervals)")
plt.ylabel("volume(vehicle/period)")
plt.ylim(0, 800)
plt.show()

plt.figure(figsize=(12, 6))
plt.plot(arima.predictions[test_start + 40:test_start + 140], '-+', label="ARIMA预测结果")
plt.plot(inv_yhat[test_start + 40:test_start + 140], '-.', label="LSTM预测结果")
plt.plot(hm_p[40:140], '--.', label='HM预测结果')
plt.plot(svr_test[40:140], '-x', label='SVR预测结果')
plt.plot(inv_y[test_start + 40:test_start + 140], '-', label="真实流量")
# plt.plot(dyn_combined[arima.windows_size - windows_size:], '--', color='r', label="arima-lstm-linear-5")
plt.legend(loc='upper right')
plt.xlabel("周期(15min)")
plt.ylabel("车流量(车辆数/周期）")
plt.show()

real = numpy.array(inv_y[test_start:])
arima = numpy.array(arima.predictions[test_start:])
lstm = numpy.array(inv_yhat[test_start:])
arima_5 = numpy.array(dyn_combined[10 - windows_size:])
arima_d = list()
lstm_d = list()
hm_d = list()
svr_d = list()
arima_5_d = list()
for i in range(40, 140):
    arima_d.append(math.fabs(arima[i] - real[i]))
    lstm_d.append(math.fabs(lstm[i] - real[i]))
    hm_d.append(math.fabs(hm_p[i] - real[i]))
    svr_d.append(math.fabs(svr_test[i] - real[i]))
    arima_5_d.append(math.fabs(arima_5[i] - real[i]))

plt.figure(figsize=(12, 6))
plt.plot(arima_d, '-', label="ARIMA 误差")
plt.plot(lstm_d, '-', color='r', label="LSTM 误差")
plt.plot(hm_d, '-', color='g', label="HM 误差")
plt.plot(svr_d, '-', label="SVR 误差")
# plt.plot(dyn_combined[arima.windows_size - windows_size:], '--', color='r', label="arima-lstm-linear-5")
plt.legend(loc='upper right')
plt.xlabel("周期(15min)")
plt.ylabel("车流量(车辆数/周期）")
plt.ylim(0, 250)
plt.show()

plt.figure(figsize=(12, 6))
plt.plot(inv_y[test_start + 40:test_start + 140], '-', label="真实流量")
plt.plot(dyn_combined[50 - windows_size:50 - windows_size + 100], '-x', color='red',
         label="arima-lstm-linear-7预测结果")
plt.plot(inv_yhat[test_start + 40:test_start + 140], '-.', color='g', label="LSTM预测结果")
# plt.plot(dyn_combined[arima.windows_size - windows_size:], '--', color='r', label="arima-lstm-linear-5")
plt.legend(loc='upper right')
plt.xlabel("周期(15min)")
plt.ylabel("车流量(车辆数/周期）")
plt.show()

plt.figure(figsize=(12, 6))
plt.plot(arima_5_d, '-', color='g', label="arima-lstm-linear-7 误差")
plt.plot(lstm_d, '-', color='r', label="LSTM 误差")
# plt.plot(dyn_combined[arima.windows_size - windows_size:], '--', color='r', label="arima-lstm-linear-5")
plt.legend(loc='upper right')
plt.xlabel("周期(15min)")
plt.ylabel("车流量(车辆数/周期）")
plt.ylim(0, 120)
plt.show()
