#%% Import Lib
import math

import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, \
    f1_score
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve

#%% - Import Data
df = pd.read_excel('./data/data_build_model.xlsx')
print(df.info())
print(df.shape)

#%% - Split data
# Stratified Train Test Split
X = df.drop('Converted', axis=1)
y = df['Converted']
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.33, random_state=100)
df1 = df.copy()
#Checking the shape of the created Train & Test DFs
print(" Shape of X_train is : ",X_train.shape)
print(" Shape of y_train is : ",y_train.shape)
print(" Shape of X_test is  : ",X_test.shape)
print(" Shape of y_test is  : ",y_test.shape)

#%% - Scale data
scaler = StandardScaler()
#Numerical features
num_cols=['TotalVisits', 'Total Time Spent on Website', 'Page Views Per Visit']
#Fitting scaler and transforming X_train
X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
#Transforming X_test
X_test[num_cols] = scaler.transform(X_test[num_cols])

#%% -
#Function to Optimal features base RFE
def optimal_features(min, max):
    opt = list()
    for features in range(min, max):
        log_reg = LogisticRegression()
        rfe = RFE(estimator = log_reg, n_features_to_select = features)
        rfe.fit(X_train, y_train)
        cols = X_train.columns[rfe.support_]

        # Cross Validation
        scores = cross_validate(log_reg, X_train[cols], y_train, return_train_score=True, cv=5,scoring=['accuracy'])
        opt.append((features, scores['test_accuracy'].mean()))

    opt = np.array(opt)
    return opt, opt[opt[:, 1].argmax()]

#Function to build logistic regression model
def build_logistic_model(feature_list):
    X_train_local = X_train[feature_list] # get feature list for VIF
    X_train_sm = sm.add_constant(X_train_local) # required by statsmodels
    log_model = sm.GLM(y_train,X_train_sm, family = sm.families.Binomial()).fit() # build model and learn coefficients
    return(log_model, X_train_sm) # return the model and the X_train fitted with constant

#Function to calculate VIF
def calculate_VIF(X_train):
    vif = pd.DataFrame()
    vif['Features'] = X_train.columns # Read the feature names
    vif['VIF'] = [variance_inflation_factor(X_train.values,i) for i in range(X_train.shape[1])] # calculate VIF
    vif['VIF'] = round(vif['VIF'],2)
    vif.sort_values(by='VIF', ascending = False, inplace=True)
    return(vif) # returns the calculated VIFs for all the features



def plot_roc( actual, probs ):
    fpr, tpr, thresholds = roc_curve( actual, probs,
                                              drop_intermediate = False )
    auc_score = roc_auc_score( actual, probs )
    plt.figure(figsize=(8, 6))
    plt.plot( fpr, tpr, label='ROC curve (area = %0.2f)' % auc_score )
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate or [1 - True Negative Rate]')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()

    return None

#%% -
feat_array, opt_features = optimal_features(10,  46)

plt.figure(figsize=(12, 6))
plt.plot(feat_array[:, 0], feat_array[:, 1])
plt.xlabel("Number of Features")
plt.ylabel("Cross Validated Mean Accuracy")
plt.show()

print(f"Optimal number of features to use is {opt_features[0]} which gives {opt_features[1]} accuracy.")
#%% - RFE
log_reg = LogisticRegression()
rfe = RFE(estimator = log_reg, n_features_to_select = int(opt_features[0]))
rfe.fit(X_train, y_train)
cols = X_train.columns[rfe.support_]
print(f"The columns we'll be using are:\n\n{cols}")

#%% - Model 1
features = list(cols) #  Use RFE selected variables
df.to_csv('./data/data.csv', index=False)
log_model1, X_train_sm1 = build_logistic_model(features) # Call the function and get the model and the X_train_sm for prediction
print(log_model1.summary())
print(calculate_VIF(X_train))

#%% - Model 2
features.remove('What is your current occupation_Housewife') # Remove 'Occupation_Housewife number' from RFE features list
log_model2, X_train_sm2 = build_logistic_model(features) # Call the function and get the model and the X_train_sm for prediction
print(log_model2.summary())
print(calculate_VIF(X_train[features]))

#%% - Model 3
features.remove('Lead Origin_Lead Import')
log_model3, X_train_sm3 = build_logistic_model(features) # Call the function and get the model and the X_train_sm for prediction
print(log_model3.summary())
print(calculate_VIF(X_train[features]))

#%%- Model 4
features.remove('Lead Source_Referral Sites')
log_model4, X_train_sm4 = build_logistic_model(features) # Call the function and get the model and the X_train_sm for prediction
print(log_model4.summary())
print(calculate_VIF(X_train[features]))

#%%- Model 5
features.remove('City_Tier II Cities')
log_model5, X_train_sm5 = build_logistic_model(features) # Call the function and get the model and the X_train_sm for prediction
print(log_model5.summary())
print(calculate_VIF(X_train[features]))

#%%- Model 6
features.remove('City_Other Metro Cities')
log_model6, X_train_sm6 = build_logistic_model(features) # Call the function and get the model and the X_train_sm for prediction
print(log_model6.summary())
print(calculate_VIF(X_train[features]))

#%% - Model 7
features.remove('Specialization_Business Administration')
log_model7, X_train_sm7 = build_logistic_model(features) # Call the function and get the model and the X_train_sm for prediction
print(log_model7.summary())
print(calculate_VIF(X_train[features]))

#%% - Model 8
features.remove('Specialization_Management_Specializations')
log_model8, X_train_sm8 = build_logistic_model(features) # Call the function and get the model and the X_train_sm for prediction
print(log_model8.summary())
print(calculate_VIF(X_train[features]))

#%% - Model 9
features.remove('What is your current occupation_Working Professional')
log_model9, X_train_sm9 = build_logistic_model(features) # Call the function and get the model and the X_train_sm for prediction
print(log_model9.summary())
print(calculate_VIF(X_train[features]))

#%% - Model 10
features.remove('City_Other Cities of Maharashtra')
log_model10, X_train_sm10 = build_logistic_model(features) # Call the function and get the model and the X_train_sm for prediction
print(log_model10.summary())
print(calculate_VIF(X_train[features]))

#%% - Model 11
features.remove('Last Activity_Olark Chat Conversation')
log_model11, X_train_sm11 = build_logistic_model(features) # Call the function and get the model and the X_train_sm for prediction
print(log_model11.summary())
print(calculate_VIF(X_train[features]))
#%% - Model 12
features.remove('What is your current occupation_Unemployed')
log_model12, X_train_sm12 = build_logistic_model(features) # Call the function and get the model and the X_train_sm for prediction
print(log_model12.summary())
print(calculate_VIF(X_train[features]))

#%% - Model 13
features.remove('What is your current occupation_Student')
log_model13, X_train_sm13 = build_logistic_model(features) # Call the function and get the model and the X_train_sm for prediction
print(log_model13.summary())
print(calculate_VIF(X_train[features]))

#%%
X_train = X_train[features]
X_test = X_test[features]

#%%
lr_model = LogisticRegression()
lr_model.fit(X_train,y_train)
y_train_prob = lr_model.predict_proba(X_train)[:,1]

#%% -
y_pred =lr_model.predict(X_train)
#Dataframe to store Lead Score, y_train, and predictions together
lr_pred = pd.DataFrame({'Converted': y_train, 'Convert_prob': y_train_prob, 'Lead Score': (y_train_prob*100).round(2), 'Predicted': y_pred})
print(lr_pred)

#%% - Confusion Matrix for train data
print(classification_report(y_train,y_pred))
cm = confusion_matrix(y_train,y_pred)
cm_df = pd.DataFrame(data = cm)
plt.figure(figsize = (6,5))
sns.heatmap(cm_df, fmt='g',
            yticklabels=['Not Converted', 'Converted'],
            xticklabels=['Predicted Not Converted', 'Predicted Converted'],
            annot=True,
            linewidths=.2,linecolor="k", cmap = "Blues", square=True, annot_kws={"fontsize":20})
plt.title('Confusion Matrix', fontsize=14)
print('\nOverall Accuracy Score : {:.2f}\n\n'.format(accuracy_score(y_train,y_pred)))

plt.show()

#%%
plot_roc(lr_pred['Converted'], lr_pred['Convert_prob'])

#%%
numbers = [float(x)/10 for x in range(10)]
for i in numbers:
    lr_pred[i]= lr_pred['Convert_prob'].map(lambda x: 1 if x > i else 0)
lr_pred.head()

#%% - Cutoff point
cutoff_df = pd.DataFrame(columns=['prob_thresh', 'accuracy', 'sensi', 'speci'])

num = [float(x) / 10 for x in range(10)]
for i in num:
    cm1 = confusion_matrix(lr_pred['Converted'], lr_pred[i])
    total1 = sum(sum(cm1))
    accuracy = (cm1[0, 0] + cm1[1, 1]) / total1

    speci = cm1[0, 0] / (cm1[0, 0] + cm1[0, 1])
    sensi = cm1[1, 1] / (cm1[1, 0] + cm1[1, 1])
    cutoff_df.loc[i] = [i, accuracy, sensi, speci]
print(cutoff_df)


cutoff_df.plot.line(x='prob_thresh', y=['accuracy','sensi','speci'], figsize=(8, 6))
plt.xlabel('Threshold')
plt.axvline(x=0.37, color='r', linestyle='--')
plt.show()
#%% - Create a matrix to Print the Accuracy, Sensitivity and Specificity
def lg_metrics(confusion_matrix):
    TN = confusion_matrix[0, 0]
    TP = confusion_matrix[1, 1]
    FP = confusion_matrix[0, 1]
    FN = confusion_matrix[1, 0]
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    speci = TN / (TN + FP)
    sensi = TP / (TP + FN)
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    TPR = TP / (TP + FN)
    TNR = TN / (TN + FP)
    FPR = FP / (TN + FP)
    FNR = FN / (TP + FN)
    pos_pred_val = TP / (TP + FP)
    neg_pred_val = TN / (TN + FN)

    print("Model Accuracy value is              : ", round(accuracy * 100, 2), "%")
    print("Model Sensitivity value is           : ", round(sensi * 100, 2), "%")
    print("Model Specificity value is           : ", round(speci * 100, 2), "%")
    print("Model Precision value is             : ", round(precision * 100, 2), "%")
    print("Model Recall value is                : ", round(recall * 100, 2), "%")
    print("Model True Positive Rate (TPR)       : ", round(TPR * 100, 2), "%")
    print("Model False Positive Rate (FPR)      : ", round(FPR * 100, 2), "%")
    print("Model Poitive Prediction Value is    : ", round(pos_pred_val * 100, 2), "%")
    print("Model Negative Prediction value is   : ", round(neg_pred_val * 100, 2), "%")


#%%- Cut-off of 0.37
y_train_pred = np.array([1 if i > 0.37 else 0 for i in y_train_prob])
print('\nOverall Accuracy Score : {}\n'.format(accuracy_score(y_train, y_train_pred)))
print('Train confusion matrix: \n\n {}'.format(confusion_matrix(y_train, y_train_pred)))
cm = confusion_matrix(y_train, y_train_pred)
cm_df = pd.DataFrame(data = cm)
plt.figure(figsize = (6,5))
sns.heatmap(cm_df, fmt='g',
            yticklabels=['Not Converted', 'Converted'],
            xticklabels=['Predicted Not Converted', 'Predicted Converted'],
            annot=True,
            linewidths=.2,linecolor="k", cmap = "Blues", square=True, annot_kws={"fontsize":20})
plt.title('Confusion Matrix', fontsize=14)
print('\nOverall Accuracy Score : {:.2f}\n\n'.format(accuracy_score(y_train, y_train_pred)))

plt.show()
lg_metrics(cm)
#%%
p, r, thresholds = precision_recall_curve(lr_pred['Converted'], lr_pred['Convert_prob'])
plt.figure(figsize=(8, 6))
plt.plot(thresholds, p[:-1], "g-", label = 'Precision')
plt.plot(thresholds, r[:-1], "r-", label = 'Recall')
plt.axvline(x=0.44, color='r', linestyle='--')
plt.xlabel('Threshold')
plt.legend(fontsize = 12)
plt.show()
#%% - Cut-off of 0.44
y_train_pred = np.array([1 if i > 0.44 else 0 for i in y_train_prob])
print('\nOverall Accuracy : {}%\n'.format((accuracy_score(y_train, y_train_pred)*100)))
print('Train confusion matrix: \n\n {}'.format(confusion_matrix(y_train, y_train_pred)))
cm = confusion_matrix(y_train, y_train_pred)
cm_df = pd.DataFrame(data = cm)
plt.figure(figsize = (6,5))
sns.heatmap(cm_df, fmt='g',
            yticklabels=['Not Converted', 'Converted'],
            xticklabels=['Predicted Not Converted', 'Predicted Converted'],
            annot=True,
            linewidths=.2,linecolor="k", cmap = "Blues", square=True, annot_kws={"fontsize":20})
plt.title('Confusion Matrix', fontsize=14)
print('\nOverall Accuracy Score : {:.2f}\n\n'.format(accuracy_score(y_train, y_train_pred)))

plt.show()
lg_metrics(cm)
#%%
#classification report
print('\nIn-sample Precision Score   : {}%\n'.format((precision_score(y_train, y_train_pred)*100).round(2)))
print('In-sample Recall Score      : {}%\n'.format((recall_score(y_train, y_train_pred)*100).round(2)))
print('In-sample F-measure         : {}%\n'.format((f1_score(y_train, y_train_pred)*100).round(2)))

#%%
y_prob = lr_model.predict_proba(X_test)[:,1]
#Dataframe to store Lead Score, y_test, and predictions together
lr_test = pd.DataFrame({'Converted': y_test, 'Convert_prob': y_prob, 'Lead Score': (y_prob*100).round(2)})

lr_test['final_predicted'] = lr_test['Convert_prob'].map(lambda x: 1 if x > 0.37 else 0)
print(lr_test.head())
#%% - Confusion Matrix for test data
cm = confusion_matrix(lr_test['Converted'], lr_test['final_predicted'])
cm_df = pd.DataFrame(data = cm)

plt.figure(figsize = (6,5))
sns.heatmap(cm_df, fmt='g',
            yticklabels=['Not Converted', 'Converted'],
            xticklabels=['Predicted Not Converted', 'Predicted Converted'],
            annot=True,
            linewidths = 0.2, linecolor="k", cmap = "Blues", square=True, annot_kws={"fontsize":20})
plt.title('Confusion Matrix', fontsize=14)
print('\nOverall Accuracy : {}%\n\n'.format((accuracy_score(lr_test['Converted'], lr_test['final_predicted'])*100)))
plt.show()
lg_metrics(cm)
#%%
#classification report
print('\nTest Precision Score   : {}%\n'.format((precision_score(lr_test['Converted'], lr_test['final_predicted'])*100).round(2)))
print('Test Recall Score      : {}%\n'.format((recall_score(lr_test['Converted'], lr_test['final_predicted'])*100).round(2)))
print('Test F-measure         : {}%\n'.format((f1_score(lr_test['Converted'], lr_test['final_predicted'])*100).round(2)))

# Assuming lr_model.feature_names_in_ and lr_model.coef_ are defined somewhere
#%%
param_df = pd.DataFrame({'Variable': lr_model.feature_names_in_, 'Coefficient': lr_model.coef_.reshape(-1)}).reset_index()
param_df = param_df.sort_values('Coefficient')
intercept = lr_model.intercept_
print(intercept)
print(param_df)
#%%
def prediction_func(model, features):

    inter = model.intercept_
    coef = model.coef_

    # Calculate the linear combination of features and coefficients
    x = inter + np.dot(features, coef.flatten())  # Flatten coef array if necessary
    prob_conversion = 1 / (1 + np.exp(-x))

    return prob_conversion
#%%
#Do Not Email, TotalVisits,Total Time Spent on Website
#Page Views Per Visit,A free copy,Origin_Landing
#Origin_Lead Add,Source_Olark Chat, Source_Welingak
#Activity_Email Link,Activity_Email Opened,Activity_Other,Activity_Page
#Activity_SMS Sent, Activity_Unreachable, Activity_Unsubscribed,Specialization_Other
features = np.array([1,0,5,
                     2,0,1,
                     0,0,0,
                     0,0,0,0,
                     0,1,0,1])

# Predict probability of conversion for the given set of features using the trained model
predicted_probability = prediction_func(lr_model, features)
print("Predicted probability of conversion:", predicted_probability)



