#%% - Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from pmdarima.arima import auto_arima
from statsmodels.tsa.arima.model import ARIMA
import itertools
import seasonal
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.filters.hp_filter import hpfilter
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import warnings
warnings.filterwarnings("ignore")

#%% - Configs
plt.rcParams['figure.figsize'] = (10,8)
plt.rcParams['figure.dpi'] = 150
plt.rcParams['font.size'] = 16

#%% - Import dataset
df = pd.read_excel("./data/VIC_2007_2024.xlsx", index_col="Date", parse_dates=True)
df_hw = df
print(df.info)

# ------------------------------------------------------------------------------------------- #
#                                         ARIMA MODEL                                         #
# ------------------------------------------------------------------------------------------- #

#%%#%% - Split data
df_close = np.log(df["Close_price"])
train_data, test_data = df_close[:int(len(df_close)*0.8)], df_close[int(len(df_close)*0.8):]
plt.plot(train_data,'blue',label='Train Data')
plt.plot(test_data,'red',label='Test Data')
plt.xlabel("Date")
plt.ylabel("Close Price")
plt.show()

#%% - Calculate p, d, q automatically using Auto ARIMA
stepwise_fit = auto_arima(train_data, trace=True, suppress_warnings=True)
print(stepwise_fit.summary())
stepwise_fit.plot_diagnostics(figsize=(15, 8))
plt.show()

#%% - Create ARIMA model
model_ARIMA = ARIMA(train_data, order=(1,1,0), trend='t') #(1,1,2) là kết quả chạy ARIMA ở câu trên
fitted_ARIMA = model_ARIMA.fit() #disp=-1 bị deprecated
print(fitted_ARIMA.summary())

#%% - Forecast using ARIMA Model
fc = fitted_ARIMA.get_forecast(len(test_data))
fc_values = fc.predicted_mean
fc_values.index = test_data.index
conf = fc.conf_int(alpha=0.05) #95% conf
lower_series = conf['lower Close_price'] #Cận dưới của cột "Close"
lower_series.index = test_data.index
upper_series = conf['upper Close_price'] #Cận trên của cột "Close"
upper_series.index = test_data.index

#%% - Evaluate ARIMA model performance
mse_ARIMA = mean_squared_error(test_data, fc_values)
print('Test MSE: %.3f' % mse_ARIMA)
rmse_ARIMA = np.sqrt(mse_ARIMA)
print('Test RMSE: {:.3f}'.format(rmse_ARIMA))
mape_ARIMA = np.mean(np.abs(fc_values - test_data) / np.abs(test_data))
print('MAPE:', mape_ARIMA)

#%% - Calculate RMSE for baseline
baseline_prediction = np.full_like(test_data, train_data.mean()) # or median
baseline_rmse = np.sqrt(mean_squared_error(test_data,baseline_prediction))

#%% - Plot actual vs predicted values
plt.figure(figsize=(16, 10), dpi=150)
plt.plot(train_data, label="Training data")
plt.plot(test_data, color="orange", label="Actual stock closing price")
plt.plot(fc_values, color="red", label="Predicted stock closing price")
plt.fill_between(lower_series.index, lower_series, upper_series, color="b", alpha=.10) #alpha là độ trong suốt
plt.title("Stock price prediction")
plt.xlabel("Time")
plt.ylabel("Stock closing price")
plt.legend()
plt.show()

#%% - Visualize RMSE comparision
plt.figure(figsize=(16, 10))
plt.bar(['ARIMA Model', 'Baseline'], [rmse_ARIMA, baseline_rmse], color=['blue','green'])
plt.title('Root Mean Squared Error (RMSE) Comparision')
plt.ylabel('RMSE')
plt.show()

print('ARIMA Model RMSE: {:.2f}'.format(rmse_ARIMA))
print('Baseline RMSE: {:.2f}'.format(baseline_rmse))


# ------------------------------------------------------------------------------------------- #
#                                  HOLT-WINTERS MODEL                                         #
# ------------------------------------------------------------------------------------------- #

#%%-
df_hw.drop(['Auction_weight','Auction_price','Change','Put_through_price','Put_through_weight','Adjusted_price','Open_price','Highest_price','Lowest_price'], axis=1, inplace=True)
print(df.head(4))

#%% - Plot
ax = df_hw['Close_price'].plot()
ax.autoscale(axis='x', tight=True)
ax.set(ylabel='Average Monthly Sales Close Price')
plt.show()

#%% - Nhóm dữ liệu theo tháng và tính trung bình
df_hw = df_hw.resample('M').mean()
print(df_hw.info())
print(df_hw.head(4))

#%% - Tính chu kỳ và xu hướng của chuỗi thời gian 'Close_price' sau khi áp dụng bộ lọc HP (Hodrick-Prescott).
close_price_cycle, close_price_trend = hpfilter(df_hw['Close_price'], lamb=1200)
print(close_price_cycle)

#%% - Kiểm tra kiểu dữ liệu của sales_cycle
type(close_price_cycle)
print(close_price_trend)

#%% -
df_hw['Trend'] = close_price_trend
print(df_hw.head(4))

#%% - Plot xu hướng (Trend)
df_hw[['Close_price', 'Trend']].plot().autoscale(axis='x', tight=True)
plt.legend()
plt.show()

#%% - Plot chu kỳ (Cycle)
df_hw['Cyclic'] = close_price_cycle

ax = df_hw[['Close_price', 'Trend', 'Cyclic']].plot()
ax.set_ylabel('Prices')
ax.autoscale(axis='x', tight=True)
plt.legend()
plt.show()

#%% - Seasonal Decompose
decompose_result = seasonal_decompose(df_hw['Close_price'], model="additive", period=30)
decompose_result.plot()
plt.show()

#%% - Chia tập dữ liệu huấn luyện (train) và kiểm thử (test)
df_close_hw = np.log(df_hw['Close_price']) #log để lấy Logarit do lớn quá
train_data_hw, test_data_hw = df_close_hw[:int(len(df_close_hw)*0.8)], df_close_hw[int(len(df_close_hw)*0.8):] #Lấy 80% data để train và 20% để test
plt.plot(train_data_hw, 'blue', label='Train data')
plt.plot(test_data_hw, 'red', label='Test data')
plt.xlabel('Date')
plt.ylabel('Close prices')
plt.legend()
plt.show()

#%% - Hyperparameter training
def test_optimizer(train, test, abg, trend_mode='mul', seasonal_mode = 'mul', seasonal_period=12):
    best_alpha, best_beta, best_gamma, best_mae = None, None, None, float("inf")
    step = len(test)

    for comb in abg:
        tes_model = ExponentialSmoothing(train, trend=trend_mode, seasonal=seasonal_mode, seasonal_periods=seasonal_period).\
            fit(smoothing_level=comb[0], smoothing_trend=comb[1], smoothing_seasonal=comb[2])
        y_pred = tes_model.forecast(step)
        mae = mean_absolute_error(test, y_pred)
        if mae < best_mae:
            best_alpha, best_beta, best_gamma, best_mae = comb[0], comb[1], comb[2], mae
        print([round(comb[0], 2), round(comb[1], 2), round(comb[2], 2), round(mae, 2)])

    print("best_alpha:", round(best_alpha, 2), "best_beta:", round(best_beta, 2), "best_gamma:", round(best_gamma, 2),
          "best_mae:", round(best_mae, 4))

    return best_alpha, best_beta, best_gamma, best_mae

#%% -
alphas = betas = gammas = np.arange(0, 1, 0.2)
abg = list(itertools.product(alphas, betas, gammas))

#%% -
best_alpha, best_beta, best_gamma, best_mae = test_optimizer(train_data_hw,test_data_hw, abg)

#%% - Triple Exponential Smoothing - Multiplicative
model_fit = ExponentialSmoothing(train_data_hw, trend='mul', seasonal='mul', seasonal_periods=12).fit(smoothing_level=best_alpha, smoothing_trend=best_beta, smoothing_seasonal=best_gamma)

#%% - Predict test_data
predictions = model_fit.forecast(len(test_data_hw))

#%% - Plot result
plt.plot(train_data_hw, 'blue', label='Train data')
plt.plot(test_data_hw, 'red', label='Test data')
plt.plot(predictions, 'green', label='Predictions')
plt.xlabel('Date')
plt.ylabel('Close prices')
plt.legend()
plt.show()

#%% -
mse_hw = mean_squared_error(test_data_hw, predictions)
print('Test MSE: %.3f' % mse_hw)

rmse_hw = np.sqrt(mse_hw)
print('Test RMSE: {:.3f}'.format(rmse_hw))

mape_hw = np.mean(np.abs(predictions - test_data_hw) / np.abs(test_data_hw))
print('MAPE:', mape_hw)