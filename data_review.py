import math
import matplotlib.pyplot as plt
from pandas import read_csv
from pandas import datetime


def parser(x):
    return datetime.strptime(x, '%Y-%m-%d %H:%M:%S')


def mean_a_p_e(y_true, y_pred):
    sum = 0
    for i in range(len(y_true)):
        sum += math.fabs(y_pred[i] - y_true[i]) / y_true[i]
    return sum / len(y_true)


series = read_csv('/Users/daihanru/Desktop/arima-lstm/DataSet/FEB15.csv', usecols=[2, 3], header=0, parse_dates=[0],
                  index_col=0,
                  squeeze=True,
                  date_parser=parser)
X = series.values[0:1000]
plt.plot(X, '-', label="real flow")
plt.legend(loc='upper right')
plt.xlabel("period(15-minute intervals)")
plt.ylabel("volume(vehicle/period)")
# plt.ylim(0, 800)
plt.show(figsize=(12, 6))
