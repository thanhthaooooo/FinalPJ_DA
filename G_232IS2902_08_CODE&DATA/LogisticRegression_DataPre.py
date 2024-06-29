#%% - Load Library
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as mno
import warnings
warnings.filterwarnings("ignore")

#%% - Read dataset
df = pd.read_csv('./data/Leads.csv')
print(df.shape)
print(df.info())
#Statistical summary
print(df.describe())

#%% - Check duplicated values
df.duplicated().any()
# Replace "Select" to NaN
df.replace({'Select': np.nan}, inplace = True)
# Drop 'Prospect ID','Lead Number','Last Notable Activity'
df.drop(['Prospect ID','Lead Number','Last Notable Activity'], axis = 1, inplace = True)

#%% - Check NaN value
total = df.isnull().sum().sort_values(ascending = False)
percent = (df.isnull().sum()/df.isnull().count()*100).round(2).sort_values(ascending = False)
null_df = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

#Visualizing columns with missing values
plt.figure(figsize = (18,8))
sns.heatmap(df.isna(),cbar = False)
plt.tight_layout()
plt.show()
#%%
# Dropping columns with more than 40% missing values
missing_cols = [col for col in df.columns if (df[col].isna().sum()/df.shape[0]*100) > 40]
df.drop('Tags',axis=1,inplace=True)
df.drop(missing_cols, axis = 1, inplace = True)
#Show variable have null values, again
x = (df.isna().sum()/df.shape[0]*100)
x[x > 0].sort_values(ascending = False)

#%% - City
print(df['City'].describe())
df['City'].value_counts(dropna = False, normalize = True).mul(100).round(2).plot(kind = 'bar', figsize = [12,8])
plt.show()

df['City'] = df['City'].replace(np.nan, 'Mumbai')
df['City'].value_counts(dropna = False, normalize = True).mul(100).round(2).plot(kind = 'bar', figsize = [12,8])
plt.show()

#%% - Specialization
print(df['Specialization'].describe())
df['Specialization'].value_counts(dropna = False, normalize = True).mul(100).round(2).plot(kind = 'bar', figsize = [12,8])
plt.show()

df['Specialization'].fillna('Other Specialization', inplace = True)
df['Specialization'].value_counts(dropna = False, normalize = True).mul(100).round(2).plot(kind = 'bar', figsize = [12,8])
plt.show()

#%% - What matters most to you in choosing a course
print(df['What matters most to you in choosing a course'].describe())
df['What matters most to you in choosing a course'].value_counts(dropna = False, normalize = True).mul(100).round(2).plot(kind = 'bar', figsize = [12,8])
plt.show()

df['What matters most to you in choosing a course'] = df['What matters most to you in choosing a course'].replace(np.nan,'Better Career Prospects')
df['What matters most to you in choosing a course'].value_counts(dropna = False, normalize = True).mul(100).round(2).plot(kind = 'bar', figsize = [12,8])
plt.show()

#%% - What is your current occupation
print(df['What is your current occupation'].describe())
df['What is your current occupation'].value_counts(dropna = False, normalize = True).mul(100).round(2).plot(kind = 'bar', figsize = [12,8])
plt.show()

df['What is your current occupation'].fillna('Unemployed', inplace = True)
df['What is your current occupation'].value_counts(dropna = False, normalize = True).mul(100).round(2).plot(kind = 'bar', figsize = [12,8])
plt.show()

#%% - Country
print(df['Country'].describe())
df['Country'].value_counts(dropna = False, normalize = True).mul(100).round(2).plot(kind = 'bar', figsize = [12,8])
plt.show()

df['Country'] = df['Country'].replace(np.nan, 'India')
df['Country'].value_counts(dropna = False, normalize = True).mul(100).round(2).plot(kind = 'bar', figsize = [12,8])
plt.show()

#%% - TotalVisits
print(df['TotalVisits'].describe())
df['TotalVisits'].value_counts(dropna = False, normalize = True).mul(100).round(2).plot(kind = 'bar', figsize = [12,8])
plt.show()

m = df['TotalVisits'].median()
df['TotalVisits'].fillna(m, inplace = True)

#%% - Page Views Per Visit
print(df['Page Views Per Visit'].describe())
df['Page Views Per Visit'].value_counts(dropna = False, normalize = True).mul(100).round(2).plot(kind = 'bar', figsize = [12,8])
plt.show()

m = df['Page Views Per Visit'].median()
df['Page Views Per Visit'].fillna(m, inplace = True)

#%% - Last Activity
print(df['Last Activity'].describe())
df['Last Activity'].value_counts(dropna = False, normalize = True).mul(100).round(2).plot(kind = 'bar', figsize = [12,8])
plt.show()

df['Last Activity'].fillna('Email Opened', inplace = True)
df['Last Activity'].value_counts(dropna = False, normalize = True).mul(100).round(2).plot(kind = 'bar', figsize = [12,8])
plt.show()

#%% - Lead Source
print(df['Lead Source'].describe())
df['Lead Source'].value_counts(dropna = False, normalize = True).mul(100).round(2).plot(kind = 'bar', figsize = [12,8])
plt.show()

df['Lead Source'] = df['Lead Source'].replace(np.nan,'Google')
df['Lead Source'] = df['Lead Source'].replace('google','Google') #replacing google with Google
df['Lead Source'].value_counts(dropna = False, normalize = True).mul(100).round(2).plot(kind = 'bar', figsize = [12,8])
plt.show()

#Checking for missing values again
df.isna().any()





#Outlier Analysis
#%% Function for visualizing numerical variables, box plot
def box_plot(x):
    ax = sns.boxplot(y = df[x], color = 'darkcyan', showfliers = True, showmeans = True,
                     meanprops={"marker":"s","markerfacecolor":"white", "markeredgecolor":"crimson"})
    ax.set_ylabel('')
    ax.set_title('{}'.format(x), fontsize = 14, fontweight = 'bold', pad = 5)
    ax.patch.set_edgecolor('black')
    ax.patch.set_linewidth(1.5)

#%%
plt.figure(figsize = [10,8])
num_cols = [col for col in df.select_dtypes(include = np.number).columns if col != 'Converted']
for i in range(len(num_cols)):
    plt.subplot(1,len(num_cols),i+1)
    box_plot(num_cols[i])
plt.tight_layout()
plt.show()
#%% - Remove 5% high value
lim1 = df['TotalVisits'].quantile(0.995)
lim2 = df['Page Views Per Visit'].quantile(0.995)
df = df[~((df['TotalVisits'] > lim1) | (df['Page Views Per Visit'] > lim2))]
df.shape





# EDA
#%% - Analysing Variables
cat_cols = [col for col in df.select_dtypes(exclude = np.number).columns]
var_df = df[cat_cols].nunique().sort_values()
#Dropping the columns have only one level
x = df[cat_cols].nunique()
cols_to_drop = list(x[x < 2].index)
df.drop(cols_to_drop, axis = 1, inplace = True)
cat_cols = list(set(cat_cols) - set(cols_to_drop))

#%% - Binary catagorical variables
x = df[cat_cols].nunique()
binary_cat_cols = list(x[x == 2].index)

#Visualizing binary catagorical variables
plt.figure(figsize=[17, 12])
rows = 3
cols = 3
for i, col in enumerate(binary_cat_cols):
    ax = plt.subplot(rows, cols, i + 1)
    sns.countplot(data=df,hue = "Converted", x=col, palette='deep', ax=ax)
    plt.xticks(rotation=90)
    plt.title('{}'.format(col), fontsize=15, fontweight='bold', pad=5)
    plt.xlabel(col, fontsize=14)
    plt.ylabel('Count', fontsize=14)
    ax.patch.set_edgecolor('black')
    ax.patch.set_linewidth(1.5)
plt.tight_layout()
plt.show()

for k in binary_cat_cols:
    print('{}\n'.format(df[k].value_counts()))

#%% - Dropping the columns
df.drop(['Do Not Call',
         'Search',
         'Newspaper Article',
         'X Education Forums',
         'Newspaper',
         'Digital Advertisement',
         'Through Recommendations'], axis = 1, inplace = True)

#%% - Compare variable with 'Converted'
sns.countplot(x = "Lead Origin", hue = "Converted", data = df)
plt.show()
#%%
fig, axs = plt.subplots(figsize = (15,7.5))
sns.countplot(x = "Lead Source", hue = "Converted", data = df)
plt.show()
#%%
df['Lead Source'] = df['Lead Source'].replace(['Click2call',
                                               'Live Chat',
                                               'NC_EDM',
                                               'Pay per Click Ads',
                                               'Press_Release',
                                               'Social Media',
                                               'WeLearn',
                                               'bing',
                                               'blog',
                                               'testone',
                                               'welearnblog_Home',
                                               'youtubechannel'], 'Others')
sns.countplot(x = "Lead Source", hue = "Converted", data = df)
plt.show()
#%%
fig, axs = plt.subplots(figsize = (15,5))
sns.countplot(x = "Last Activity", hue = "Converted", data = df)
plt.show()
#%%
df['Last Activity'] = df['Last Activity'].replace(['Had a Phone Conversation',
                                                   'View in browser link Clicked',
                                                   'Visited Booth in Tradeshow',
                                                   'Approached upfront',
                                                   'Resubscribed to emails',
                                                   'Email Received',
                                                   'Email Marked Spam'], 'Other Activity')
sns.countplot(x = "Last Activity", hue = "Converted", data = df)
plt.show()
#%%
fig, axs = plt.subplots(figsize = (15,5))
sns.countplot(x = "Country", hue = "Converted", data = df)
plt.show()
df.drop(['Country'], axis = 1, inplace = True)
#%%
fig, axs = plt.subplots(figsize = (15,10))
sns.countplot(x = "Specialization", hue = "Converted", data = df)
plt.show()
#combining Management Specializations because they show similar trends
df['Specialization'] = df['Specialization'].replace(['Finance Management','Human Resource Management',
                                                           'Marketing Management','Operations Management',
                                                           'IT Projects Management','Supply Chain Management',
                                                            'Healthcare Management','Hospitality Management',
                                                           'Retail Management'] ,'Management_Specializations')

#%%
fig, axs = plt.subplots(figsize = (15,10))
sns.countplot(x = "Specialization", hue = "Converted", data = df)
plt.show()
#%%
fig, axs = plt.subplots(figsize = (10,5))
sns.countplot(x = "What is your current occupation", hue = "Converted", data = df)
plt.show()
#%%
fig, axs = plt.subplots(figsize = (10,5))
sns.countplot(x = "What matters most to you in choosing a course", hue = "Converted", data = df)
plt.show()
df.drop(['What matters most to you in choosing a course'], axis = 1, inplace = True)

#%%
fig, axs = plt.subplots(figsize = (10,5))
sns.countplot(x = "City", hue = "Converted", data = df)
plt.show()

#%% - Analysing Numerical Variables
def box_plot2(x, y, p='deep'):
    ax = sns.boxplot(x=x, y=y, data=df, palette=p, showfliers=True, showmeans=True,
                     meanprops={"marker":"s", "markerfacecolor":"white", "markeredgecolor":"crimson"})
    ax.set_title('Distribution of "{}"'.format(y), fontsize=15, fontweight='bold', pad=5)
    ax.patch.set_edgecolor('black')
    ax.patch.set_linewidth(1.5)
plt.figure(figsize=[18, 8])
#%%
for i in range(len(num_cols)):
    plt.subplot(1, 3, i+1)
    box_plot2('Converted', num_cols[i], p='BuGn')
plt.tight_layout()
plt.show()




#Data preprocessing
#%%
df.shape

#%% - Chuyển biến nhị phân
varlist =  ['Do Not Email','A free copy of Mastering The Interview']
def binary_map(x):
    return x.map({'Yes': 1, "No": 0})

df[varlist] = df[varlist].apply(binary_map)

#%% - Chuyển biến phân loại
dummy = pd.get_dummies(df[['Lead Origin', 'Lead Source', 'Last Activity', 'Specialization','What is your current occupation',
                       'City']], drop_first=True,dtype=int)

# Droping the original columns after dummy variable creation
df.drop(['Lead Origin', 'Lead Source', 'Last Activity', 'Specialization','What is your current occupation',
            'City'], axis=1, inplace = True)

#merging dataframe with dummy
df = pd.concat([df, dummy], axis=1)

print(df.shape)
#%%
plt.figure(figsize = [22,18])
sns.heatmap(df.corr(), annot = False, cmap = 'coolwarm', linecolor = 'w', linewidth = 0.2)
plt.xticks(fontsize = 14)
plt.yticks(fontsize = 14)
plt.tight_layout()
plt.show()

#%%
conv_corr = df.corr()
conv_corr_unstacked = conv_corr.abs().unstack().sort_values(kind="quicksort")
check = conv_corr.where(np.triu(np.ones(conv_corr.shape), k=1).astype(np.bool_)).stack().sort_values(ascending=False).head(10)
print(check)
# Corr > 0.8
df.drop(['Lead Source_Facebook', 'Lead Source_Reference'],axis=1, inplace = True)

#%%
#Calculate the correlation matrix between the independent variable and the target variable
correlation_matrix = df.corr()
correlation_with_target = correlation_matrix['Converted'].drop('Converted')
#Highest Corr
highest_correlation_variable = correlation_with_target.abs().idxmax()
highest_correlation_value = correlation_with_target.abs().max()
print("Biến có mối tương quan cao nhất với biến mục tiêu:", highest_correlation_variable)
print("Mối tương quan cao nhất:", highest_correlation_value)
#%%
df.to_excel('./data/data_build_model.xlsx', index=False)
