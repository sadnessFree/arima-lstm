import pandas
import numpy
import math
import matplotlib
import matplotlib.pyplot as plt
from pandas import read_csv
from pandas import datetime
from sklearn import svm
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from pandas import concat
from pandas import DataFrame


def mean_a_p_e(y_true, y_pred):
    sum = 0
    for i in range(len(y_true)):
        sum += math.fabs(y_pred[i] - y_true[i]) / y_true[i]
    return sum / len(y_true)


def parser(x):
    return datetime.strptime(x, '%Y-%m-%d %H:%M:%S')


series = read_csv('/Users/daihanru/Desktop/arima-lstm/DataSet/FEB15-2.csv', usecols=[2, 3], header=0, parse_dates=[0],
                  index_col=0,
                  squeeze=True,
                  date_parser=parser)
X = series.values
X = X.astype('float32')

train_start = 4500
train_end = 6000

test_start = 6409
test_end = 6600

svr_train, svr_test = X[train_start:train_end], X[test_start - 1:test_end - 1]
svr_real = numpy.array(X[test_start:test_end]).reshape(-1, 1)
svr_train = numpy.array(svr_train).reshape(-1, 1)
svr_test = numpy.array(svr_test).reshape(-1, 1)
model = svm.SVR()
model.fit(svr_train[0:train_end - train_start - 1], svr_train[1:train_end - train_start])
p = model.predict(svr_test)

mse = mean_squared_error(svr_real, svr_test)
rmse = math.sqrt(mean_squared_error(svr_real, svr_test))
mae = mean_absolute_error(svr_real, svr_test)
mape = mean_a_p_e(svr_real, svr_test)
print('SVR Test MAE:%.3f MSE: %.3f RMSE:%.3f MAPE:%.3f' % (mae, mse, rmse, mape))
plt.plot(svr_real, '-', label="real flow")
plt.plot(svr_test, '--', color='red', label="SVR")
plt.legend(loc='upper right')
plt.xlabel("period(15-minute intervals)")
plt.ylabel("volume(vehicle/period)")
plt.ylim(0, 800)
plt.show(figsize=(12, 6))

datStart = 31

matplotlib.rcParams['font.family'] = 'SimHei'

plt.figure(figsize=(12, 6))
plt.plot(X[datStart:datStart + 96*7], '-', color='r', label="真实流量")
# plt.plot(X[datStart+96:datStart+96+96], '-',  label="2月26日")
# plt.plot(X[datStart+96+96:datStart+96+96+96], '-',  label="2月27日")
plt.legend(loc='upper right')
plt.xlabel("周期(15min)")
plt.ylabel("车流量(车辆数/周期）")
# plt.ylim(0, 400)
plt.show()
