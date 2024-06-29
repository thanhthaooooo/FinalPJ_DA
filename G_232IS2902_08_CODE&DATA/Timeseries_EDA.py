#%% - Import Lib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller,kpss
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
import warnings
warnings.filterwarnings("ignore")

#%% - Import Data
df = pd.read_excel('./data/VIC_2007_2024.xlsx')
print(df.info())
print(df.describe())

#%%
plt.rcParams["figure.figsize"] = (10,8)
plt.rcParams["figure.dpi"] = 150
plt.rcParams["font.size"] = 14
#%% - Check any null values present
df.isnull().sum()

#%% Set index and parse dates, drop
df.set_index("Date", inplace=True)
df.index = pd.to_datetime(df.index)
df['Vol'] = df['Auction_weight'] + df['Put_through_weight']
df.drop(['Auction_weight','Auction_price','Change','Put_through_price','Put_through_weight','Adjusted_price'], axis=1, inplace=True)
#%%
sns.heatmap(df.corr(),cmap='Blues',annot=True)
plt.tight_layout()
plt.show()

#%%
data2 = df.copy()
data2['Open-High'] = df['Open_price']-df['Highest_price']
data2['Open-Low'] = df['Open_price'] - df['Lowest_price']
data2['Close-High'] = df['Close_price']-df['Highest_price']
data2['Close-Low'] = df['Close_price'] - df['Lowest_price']
data2['High-Low'] = df['Highest_price'] - df['Lowest_price']
data2['Open-Close'] = df['Open_price'] - df['Lowest_price']
data2 = data2.drop(['Open_price','Highest_price','Lowest_price','Close_price'],axis=1)
#%%
sns.heatmap(data2.corr(),cmap='Blues',annot=True)
plt.tight_layout()
plt.show()
df.drop(['Vol','Highest_price','Lowest_price','Open_price'], axis=1, inplace=True)


#%% -  Initial assessment
plt.plot(df['Close_price'])
plt.xlabel("Date")
plt.ylabel("Close price")
plt.show()

#%%#%% - Slip data
df_close = np.log(df["Close_price"])
train_data, test_data = df_close[:int(len(df_close)*0.8)], df_close[int(len(df_close)*0.8):]
plt.plot(train_data,'blue',label='Train Data')
plt.plot(test_data,'red',label='Test Data')
plt.xlabel("Date")
plt.ylabel("Close Price")
plt.show()

#%% - Decompose
rolmean = train_data.rolling(12).mean()
rolstd = train_data.rolling(12).std()
plt.plot(train_data,'blue', label="Original")
plt.plot(rolmean,'red',label="Rolling Mean")
plt.plot(rolstd,'Orange',label="Rolling STD")
plt.legend()
plt.show()

#%%
decompose_results = seasonal_decompose(train_data,model="multiplication", period=30)
decompose_results.plot()
plt.show()

#%% - Test the stationarity of the data using ADF or KPSS
def adf_test(data):
    indices = ["ADF: Test statistic","p value","# of Lags","# of Observations"]
    test = adfuller(data, autolag="AIC")
    results = pd.Series(test[:4], index=indices)
    for key, value in test[4].items():
        results[f"Critical value ({key})"] = value
    if results[1] <=0.05:
        print('Reject the null hyposthesis (H0), the data is stationary')
    else:
        print('Fail to reject the null hyposthesis (H0), the data is non-stationary')
    return results

def kpss_test(data):
    indices = ["KPSS: Test statistic", "p value", "# of Lags"]
    test = kpss(data)
    results = pd.Series(test[:3], index=indices)
    for key, value in test[3].items():
        results[f"Critical value ({key})"] = value
    if results[1] >= 0.05:
        print('Reject the null hyposthesis (H0), the data is stationary')
    else:
        print('Fail to reject the null hyposthesis (H0), the data is non-stationary')
    return results

print(adf_test(train_data))
print("------------"*5)
print(kpss_test(train_data))

#%% Chuyển đổi dữ liệu sang chuỗi dừng -> sai phân bậc 1 rồi kiểm tra dừng nếu không thif sai phân bậc 2
diff = train_data.diff(1).dropna()
# Biểu đồ thể hiện dữ liệu ban đầu và sau khi lấy sai phân
fig, ax = plt.subplots(2, sharex="all")
train_data.plot(ax=ax[0],title="Gía đóng cửa")
diff.plot(ax=ax[1], title="Gía đóng cửa")
plt.show()

#%% - Kiểm tra lại tính dừng dữ liệu sau khi lấy sai phân
print(adf_test(diff))
print("-----------"*5 )
print(kpss_test(diff))
#%%
plot_pacf(diff)
plt.show() # Xác định tham số p cho ARIMA p=0 hoặc 1

#%%
plot_acf(diff)
plt.show() #q = 0,1

#%%
def get_aic(orders, data):
    for order in orders:
        try:
            model = ARIMA(data, order=order,trend='t').fit()
            model_name = 'ARIMA({},{},{})'.format(order[0], order[1], order[2])
            print('{} --> AIC={}'.format(model_name, model.aic))
        except:
            print("Failed to fit model for order:", order)

#
orders = [(0, 1, 0), (0, 1, 1), (1, 1, 0), (1, 1, 1)]
print(get_aic(orders, train_data))






