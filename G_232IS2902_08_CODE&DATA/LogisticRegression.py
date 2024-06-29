#%% - Import Lib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import math
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

#%% - Import Data
df = pd.read_excel('./data/data_build_model.xlsx')
print(df.info())

#%% - Scatter chart
plt.scatter(df['Total Time Spent on Website'], df['Converted'], color="darkgreen", marker="o")
plt.xlabel('Total Time Spent on Website')
plt.ylabel('Converted')
plt.title('Relationship between Total Time Spent on Website and Conversion')
plt.show()

#%% - Slipt train/test dataset
# Stratified Train Test Split
X = df[['Total Time Spent on Website']]
y = df['Converted']
X_train, X_test, y_train, y_test = train_test_split(X, y,stratify=y,test_size=0.33, random_state=42)

#Checking the shape of the created Train & Test
print(" Shape of X_train is : ",X_train.shape)
print(" Shape of y_train is : ",y_train.shape)
print(" Shape of X_test is  : ",X_test.shape)
print(" Shape of y_test is  : ",y_test.shape)

#%% - Create model
from sklearn.linear_model import LogisticRegression
model = LogisticRegression().fit(X_train, y_train)
# logit(odds) = a+b * Total Time Spent on Website

#%%- Get results
intercept = model.intercept_
coef = model.coef_
score = model.score(X_train,y_train)
pro_matrix = model.predict_proba(X_train)

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


#%% -
y_train_prob = model.predict_proba(X_train)[:,1]
y_pred = model.predict(X_train)
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
lg_metrics(cm)

#%% - Scatter chart train data
plt.scatter(X_train,y_train, color="cyan", marker="o",label='Actual')
plt.scatter(X_train,y_pred, color="red", marker="+",label='Predict')
plt.legend()
plt.show()

#%% - ROC
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

plot_roc(lr_pred['Converted'], lr_pred['Convert_prob'])
lg_metrics(cm)

#%%
y_prob = model.predict_proba(X_test)[:,1]
y_pred = model.predict(X_test)
#Dataframe to store Lead Score, y_test, and predictions together
lr_test = pd.DataFrame({'Converted': y_test, 'Convert_prob': y_prob, 'Lead Score': (y_prob*100).round(2),'Predicted': y_pred})


#%% - Confusion Matrix for test data
cm = confusion_matrix(lr_test['Converted'], lr_test['Predicted'])
cm_df = pd.DataFrame(data = cm)
plt.figure(figsize = (6,5))
sns.heatmap(cm_df, fmt='g',
            yticklabels=['Not Converted', 'Converted'],
            xticklabels=['Predicted Not Converted', 'Predicted Converted'],
            annot=True,
            linewidths = 0.2, linecolor="k", cmap = "Blues", square=True, annot_kws={"fontsize":20})
plt.title('Confusion Matrix', fontsize=14)
print('\nOverall Accuracy : {}%\n\n'.format((accuracy_score(lr_test['Converted'], lr_test['Predicted'])*100)))
plt.show()
lg_metrics(cm)

#%% - Predict
pred_values = model.predict(X_test)
pred_score = model.score(X_test, y_test)
pre_prob_matrix =model.predict_proba(X_test)
#%% - Define sigmoid
def sigmoid(x):
    return 1 / (1 + math.exp(-x))

#%% - define prediction function via sigmoid
def prediction_func(time, inter, coef):
    x = inter + coef*time
    return sigmoid(x)

#%% - Draw Sigmoid Plot
plt.scatter(X_train,y_train, color="red", marker='o')
x_test = np.linspace(0,2280)
sigs = []
inte = intercept[0]
co = coef[0][0]
for item in x_test:
    # print(prediction_func(item, inte, co))
    sigs.append(prediction_func(item,intercept[0],coef[0][0]))
plt.plot(x_test, sigs, color="g")
plt.scatter(X_test,y_test, color="b", label = 'Actual')
plt.scatter(X_test,lr_test['Predicted'], color="y", label = 'Predict')
plt.legend(loc="center right")
plt.show()

#%%
param_df = pd.DataFrame({'Variable': model.feature_names_in_, 'Coefficient': model.coef_.reshape(-1)}).reset_index()
param_df = param_df.sort_values('Coefficient')
intercept = model.intercept_
print(intercept)
print(param_df)

#%% - predict future values
pred_prob = prediction_func(1000, intercept[0], coef[0][0])
print("Predicted probability of conversion:", pred_prob)



