#%% - Library
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from plotnine import ggplot, aes, geom_boxplot, coord_flip
import warnings
warnings.filterwarnings("ignore")
#%% - Info
df = pd.read_csv("./data/BostonHousing.csv")
df.info()
#%% - Statistics
df.describe()

#%% - ## Clean Data ##
# Check duplicated data
df.duplicated().any()

#%% - Check NaN value
total = df.isnull().sum().sort_values(ascending = False)
percent = (df.isnull().sum()/df.isnull().count()*100).round(2).sort_values(ascending = False)
null_df = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

#%% - Filling the missing values
median_rn = df['rm'].median()
df['rm'] = df['rm'].fillna(median_rn)

#%% - Check Outlier
df_melted = df.melt(var_name='variable', value_name='value')
(ggplot(df_melted, aes(x='variable', y='value')) + geom_boxplot() + coord_flip()).show()

#%% - Remove Outlier
for column in df.columns:
    if column != "chas":
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1

        upper_limit = Q3 + 1.5 * IQR
        lower_limit = Q1 - 1.5 * IQR

        df[column] = df[column].apply(lambda x: upper_limit if x > upper_limit else x)
        df[column] = df[column].apply(lambda x: lower_limit if x < lower_limit else x)

#%% - Check Outlier again
df_melted = df.melt(var_name='variable', value_name='value')
(ggplot(df_melted, aes(x='variable', y='value')) + geom_boxplot() + coord_flip()).show()

#%% - Check statictis data, again
df_d = df.describe()
#%% - Data Visualization
# Plot
pos = 1
fig = plt.figure(figsize=(15,25))
for i in df.columns:
    ax = fig.add_subplot(7,2,pos)
    pos = pos + 1
    sns.distplot(df[i],ax=ax, color='#007acc')
plt.show()

#%% - Scatter plot with "medv"
sns.set(style="ticks")
fig = plt.figure(figsize=(12, 25))

base_variable = 'medv'

comparison_variables = list(df.columns)
comparison_variables.remove(base_variable)

for pos, var in enumerate(comparison_variables, 1):
    ax = fig.add_subplot(7, 2, pos)
    sns.scatterplot(x=df[base_variable], y=df[var], ax=ax, color='#007acc')
    ax.set_xlabel(base_variable)
    ax.set_ylabel(var)

plt.tight_layout()
plt.show()


#%% - Correlation heatmap
correlation_matrix = df.corr()
plt.figure(figsize=(10, 6))
sns.heatmap(correlation_matrix, cmap='Reds', annot=True)
plt.title('Correlation heatmap')
plt.show()
#%% - Correlation with "medv"
correlation = df.corr()['medv']
head = pd.DataFrame(correlation.sort_values(ascending=True))

#%% - Save
df.to_csv("./data/BostonHousing_Clean.csv", index=False)




