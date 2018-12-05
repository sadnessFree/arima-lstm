import math
import matplotlib.pyplot as plt
from pandas import read_csv
from pandas import datetime
import statsmodels.tsa.stattools as st
import seaborn as sns


def parser(x):
    return datetime.strptime(x, '%Y-%m-%d %H:%M:%S')


def mean_a_p_e(y_true, y_pred):
    sum = 0
    for i in range(len(y_true)):
        sum += math.fabs(y_pred[i] - y_true[i]) / y_true[i]
    return sum / len(y_true)


series = read_csv('/Users/daihanru/Desktop/arima-lstm/DataSet/FEB15-2.csv', usecols=[2, 3], header=0, parse_dates=[0],
                  index_col=0,
                  squeeze=True,
                  date_parser=parser)
series = series
X = series.values

start = 2
size = 200
arima_train = X[start:start + size]
print(st.adfuller(arima_train))
# order = st.arma_order_select_ic(arima_train, max_ar=5, max_ma=5, ic=['aic'], trend='nc')
# print(order)
# fig, ax = plt.subplots(figsize=(10, 8))
# ax = sns.heatmap(order["aic"],
#                  ax=ax,
#                  annot=True,
#                  fmt='.2f',
#                  )
# ax.set_title('AIC')
# plt.show()
