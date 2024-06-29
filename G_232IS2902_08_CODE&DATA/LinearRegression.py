#%% - Import library
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import warnings
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
warnings.filterwarnings("ignore")
#%% - Load data
df = pd.read_csv('./data/BostonHousing_Clean.csv')
#%% - Preparing

X = df[['lstat']]
Y = df['medv']

# - Function
def plot(ax, x_true, y_true, y_pred, data_type):
    # Plot Residuals
    ax[0].set_title(f'Residual Plot of {data_type} set', fontsize=14, fontweight='bold')
    sns.histplot((y_true - y_pred), kde=True, ax=ax[0], color='#007acc')
    ax[0].set_xlabel(f'Errors')
    # Scatters
    ax[1].scatter(x_true, y_true, label="Training Data")
    ax[1].plot(x_true, y_pred, color='red', label='Regression Line')
    ax[1].set_title(f"Linear Regression - Lstat on {data_type} data")
    ax[1].set_xlabel("Lstat")
    ax[1].set_ylabel("Medv")
    ax[1].legend()

    plt.show()

#%% - Split train/test set
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=1)

#%% - Fit model
model = LinearRegression()
model.fit(X_train, y_train)

#%% - Parameters
intercept = model.intercept_
coef = model.coef_[0]

y_pred_train = model.predict(X_train)

r2_train = metrics.r2_score(y_train, y_pred_train)
mse_train = mean_squared_error(y_train,y_pred_train)
mae_train = mean_absolute_error(y_train, y_pred_train)

#%% Plotting for train data
fig_train, ax_train = plt.subplots(1, 2, figsize=(15, 7))
plot(ax_train,X_train, y_train, y_pred_train, 'train')

#%% - Evaluation
y_pred_test = model.predict(X_test)
r2_test = metrics.r2_score(y_test, y_pred_test)
mse_test = mean_squared_error(y_test,y_pred_test)
mae_test = mean_absolute_error(y_test, y_pred_test)

#%% Plotting for test data
fig_test, ax_test = plt.subplots(1, 2, figsize=(15, 7))
plot(ax_test, X_test,y_test, y_pred_test, 'test')

#%% - Predict
future_lstat = np.array([10]).reshape(-1,1)
predicted_future_values = model.predict(future_lstat)
print(predicted_future_values)