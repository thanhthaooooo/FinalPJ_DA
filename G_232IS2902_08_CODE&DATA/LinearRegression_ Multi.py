#%% - Library
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from statsmodels.stats.outliers_influence import variance_inflation_factor
import warnings
warnings.filterwarnings("ignore")
#%% 
df= pd.read_csv('./data/BostonHousing_Clean.csv')

#%% --- CHECK INDEPENDENT_VARS ---
# Check p-value 1st
independent_vars = ['crim','zn','indus', 'chas', 'nox', 'rm', 'age', 'dis','rad', 'tax', 'ptratio', 'b', 'lstat']
X = df[independent_vars]
X = sm.add_constant(X)
y = df['medv']
model = sm.OLS(y, X).fit()
print(model.summary())
#%% - Remove "indus"
# Check p-value 2nd
independent_vars.remove('indus')
X = df[independent_vars]
X = sm.add_constant(X)
y = df['medv']
model = sm.OLS(y, X).fit()
print(model.summary())
#%% - Remove "b"
# Check p-value 3rd
independent_vars.remove('b')
X = df[independent_vars]
X = sm.add_constant(X)
y = df['medv']
model = sm.OLS(y, X).fit()
print(model.summary())

#%% - Remove "age"
# Check p-value 4th
independent_vars.remove('age')
X = df[independent_vars]
X = sm.add_constant(X)
y = df['medv']
model = sm.OLS(y, X).fit()
print(model.summary())

#%% - Check VIF 1st
vif = pd.DataFrame()
vif["Features"] = X.columns
vif["VIF Factor"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif = vif.round(1)
print(vif.T.to_string(header=False))
#%% - Remove "rad"
# Check VIF 2nd
independent_vars.remove('rad')
X = df[independent_vars]
X = sm.add_constant(X)
vif = pd.DataFrame()
vif["Features"] = X.columns
vif["VIF Factor"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif = vif.round(1)
print(vif.T.to_string(header=False))

#%% - Check p-value 5th
X = sm.add_constant(X)
model = sm.OLS(y, X).fit()
print(model.summary())

#%% Remove "Crim"
# Check p-value 6th
independent_vars.remove('crim')
X = df[independent_vars]
X = sm.add_constant(X)
model = sm.OLS(y, X).fit()
print(model.summary())

#%% - Check VIF 3rd

vif = pd.DataFrame()
vif["Features"] = X.columns
vif["VIF Factor"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif = vif.round(1)
print(vif.T.to_string(header=False))

#%%
print(independent_vars)
#%% - PREPARING
X = df[independent_vars]
y = df['medv']
#%% - Split train/test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
#%% - Creat model
model = LinearRegression()
model.fit(X_train, y_train)

#%% - The parameters
intercept = model.intercept_
for feature, coef in zip(X.columns, model.coef_):
    print(f"{feature}: {coef}")

r2_train = model.score(X_train, y_train)
y_pred_train = model.predict(X_train)
mse_train = mean_squared_error(y_train,y_pred_train)
mae_train = mean_absolute_error(y_train, y_pred_train)
#%% - Plotting for train data

# Residual Plot of train data
fig, ax = plt.subplots(1,2,figsize = (15,7))
ax[0].set_title('Residual Plot of Train samples', fontsize=14, fontweight='bold')
sns.histplot((y_train-y_pred_train), kde=True, ax = ax[0],color='#007acc')
ax[0].set_xlabel('Errors')
# - Scatter plot for train set
ax[1].set_title('Actual and Predict on Train sample', fontsize=14, fontweight='bold')
ax[1].scatter(x = y_train, y = y_pred_train,color='#007acc')
ax[1].set_xlabel('Actual')
ax[1].set_ylabel('Predicted')
plt.show()

#%% - Evaluation on test set

y_pred_test = model.predict(X_test)
r2_test = metrics.r2_score(y_test, y_pred_test)
mse_test = mean_squared_error(y_test,y_pred_test)
mae_test = mean_absolute_error(y_test, y_pred_test)

#%% - Plot test set
#Residual Plot of test data
fig, ax = plt.subplots(1,2,figsize = (15,7))
ax[0].set_title('Residual Plot of Test samples', fontsize=14, fontweight='bold')
sns.histplot((y_test-y_pred_test), kde=True, ax = ax[0],color='#007acc')
ax[0].set_xlabel('Errors')

# - Scatter plot for test set
ax[1].set_title('Actual and Predict on Test sample', fontsize=14, fontweight='bold')
ax[1].scatter(x = y_test, y = y_pred_test,color='#007acc')
ax[1].set_xlabel('Actual')
ax[1].set_ylabel('Predicted')
plt.show()

#%% - Predict
Zn = 12
Chas = 1
Nox = 0.4
Rm = 5
Dis = 7.5
Tax = 617
Ptratio = 16.25
Lstat = 3.5

predicted_future_values = model.predict([[Zn, Chas, Nox, Rm, Dis, Tax, Ptratio, Lstat]])
print(predicted_future_values)






