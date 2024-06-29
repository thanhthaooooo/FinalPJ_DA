#%% - Load Library
import pandas as pd
import numpy as np
import seaborn as sns
from plotnine import ggplot, aes, geom_boxplot, coord_flip
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.combine import SMOTEENN
import warnings
warnings.filterwarnings("ignore")
#%% - Read dataset
df = pd.read_csv('./data/bank-full.csv')
df.shape
#%%
df.info()
#%% - statistical summary of all the numerical variables
df_des = df.describe()
#%% - check for duplicate values
df.duplicated().sum()

#%% - check for null values
total = df.isnull().sum().sort_values(ascending = False)
percent = (df.isnull().sum()/df.isnull().count()*100).round(2).sort_values(ascending = False)
null_df = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

#%% - Check Outlier
# - Numerical feature
num_features=['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']

plt.figure(figsize=(12, 20), facecolor='white')
plotnumber = 1
for feature in num_features:
    ax = plt.subplot(4,2,plotnumber)
    sns.boxplot(y= df[feature], data=df)
    plt.xlabel(feature)
    plotnumber+=1
plt.show()

#%% - boxplot to show target distribution with respect numerical features
plt.figure(figsize=(12, 20), facecolor='white')
plotnumber =1
for feature in num_features:
    ax = plt.subplot(4,2,plotnumber)
    sns.boxplot(x='y',y= df[feature], data=df)
    plt.xlabel(feature)
    plotnumber+=1
plt.show()

#%% - CATEGORIGAL VALUES
cat_features= df.select_dtypes(include= object)
# - Unique values
data = []
for col in df.select_dtypes(include='object').columns:
    unique_values = df[col].unique()
    data.append([col, unique_values, len(unique_values)])
df_unique_values = pd.DataFrame(data, columns=['Column', 'Unique Values', 'Total Unique Count'])
#%% - Y
deposit_counts = df['y'].value_counts()
print(deposit_counts)
print("Yes deposit percentage:", np.round((df[df['y']=='yes'].shape[0]/df.shape[0])*100,2),"%")

sns.countplot(x='y', data=df, orient='v')
plt.show()
#%% - Month
month_counts = df['month'].value_counts()
print(month_counts)

sns.countplot(x='month', data= df, orient='v')
plt.show()
#%% - Contact
contact_counts = df['contact'].value_counts()
print(contact_counts)

sns.countplot(x='contact', data= df, orient='v')
plt.show()
#%% - Marital
marital_counts = df['marital'].value_counts()
print(marital_counts)

plt.pie(df['marital'].value_counts(), labels=df['marital'].unique(), autopct='%1.2f%%')
plt.show()
#%% - Housing Loan
housing_loan_counts = df['housing'].value_counts()
print(housing_loan_counts )
print("yes precentage for house loan:", np.round(df[df['housing']=='yes'].shape[0]/df.shape[0],3))

sns.countplot(x='housing',data=df)
plt.show()
#%% - Loan
loan_counts = df['loan'].value_counts()
print(loan_counts)

sns.countplot(x='loan', data= df, orient='v')
plt.show()
#%% - Jobs
job_counts = df['job'].value_counts()
print(job_counts)

sns.countplot(x='job', data= df, orient='v')
ax = plt.gca()
ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha='right')
plt.show()
#%% - Poutcome
poutcome_counts = df['poutcome'].value_counts()
print(poutcome_counts)

sns.countplot(x='poutcome', data= df, orient='v')
plt.show()
#%% - Default credit
default_counts = df['default'].value_counts()
print(default_counts)

sns.countplot(x='default', data= df, orient='v')
plt.show()
#%% - Education
education_counts = df['education'].value_counts()
print(education_counts)

sns.countplot(x='education', data= df, orient='v')
plt.show()
#%%
X = df.drop('y', axis='columns')
y = df['y']
# Mapping
job_mapping = {'management':7, 'technician':6, 'entrepreneur':4, 'blue-collar':5, 'unknown':0, 'retired':11, 'admin.':8, 'services':9, 'self-employed':10, 'unemployed':2, 'housemaid':3, 'student':1}
X['job'] = X['job'].map(job_mapping)
marital_mapping = {'married': 2, 'single': 1, 'divorced': 0}
X['marital'] = X['marital'].map(marital_mapping)
education_mapping = {'tertiary': 3, 'secondary': 2, 'unknown': 0, 'primary': 1}
X['education'] = X['education'].map(education_mapping)
default_mapping = {'no': 0, 'yes': 1}
X['default'] = X['default'].map(default_mapping)
housing_mapping = {'yes': 1, 'no': 0}
X['housing'] = X['housing'].map(housing_mapping)
loan_mapping = {'no': 0, 'yes': 1}
X['loan'] = X['loan'].map(loan_mapping)
contact_mapping = {'unknown': 0, 'cellular': 1, 'telephone': 2}
X['contact'] = X['contact'].map(contact_mapping)
month_mapping = {'may': 5, 'jun': 6, 'jul': 7, 'aug': 8, 'oct': 10, 'nov': 11, 'dec': 12, 'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'sep': 9}
X['month'] = X['month'].map(month_mapping)
poutcome_mapping = {'unknown': 0, 'failure': 2, 'other': 3, 'success': 1}
X['poutcome'] = X['poutcome'].map(poutcome_mapping)
y_mapping = {'no': 0, 'yes': 1}
y = y.map(y_mapping)

#%% - Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
#%% - Handle Class Imbalance
# smt = SMOTE(sampling_strategy='all')
smt = SMOTEENN(sampling_strategy='all')
X_resampled, y_resampled = smt.fit_resample(X_scaled, y)
#%%
# Chuyển dữ liệu đã được xử lý lại thành DataFrame
processed_data = pd.DataFrame(X_resampled, columns=X.columns)
processed_data['y'] = y_resampled
#%%
y_count = y_resampled.value_counts()
print('Class 0:', y_count[0])
print('Class 1:', y_count[1])
print('Proportion of class 0 is ', round(y_count[0] * 100 / (y_count[1] + y_count[0]), 2),'%')
print('Proportion of class 1 is ', round(y_count[1] * 100 / (y_count[0] + y_count[1]), 2),'%')
y_count.plot(kind='bar', title='Count (target)')
plt.xlabel('y')
plt.show()
#%%
# Chia dữ liệu thành train và test theo tỉ lệ 80/20
train_data, test_data = train_test_split(processed_data, test_size=0.2)
#%%
# Lưu dữ liệu vào file CSV
train_data.to_csv('train.csv', index=False)
test_data.to_csv('test.csv', index=False)