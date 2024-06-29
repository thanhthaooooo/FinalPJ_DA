# Importing necessary libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report, ConfusionMatrixDisplay, RocCurveDisplay
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

#%% - Configs
plt.rcParams['figure.figsize'] = (10,8)
plt.rcParams['figure.dpi'] = (200)
plt.rcParams['font.size'] = (9)

#%% - Read the data
# Đọc dữ liệu từ file train.csv và test.csv
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

X_train = train_data.drop(columns=['y'])
y_train = train_data['y']
X_test = test_data.drop(columns=['y'])
y_test = test_data['y']

#%%
### Decision Tree

param_grid_dt_0 = {'max_depth': [None,5, 10, 20,25], 'min_samples_split': [2, 5, 10]}
dt_model_0 = DecisionTreeClassifier()
dt_grid = GridSearchCV(dt_model_0, param_grid_dt_0, cv=5)
dt_grid.fit(X_train, y_train)

# Extract results
results_dt_0 = dt_grid.cv_results_
results_df_dt_0 = pd.DataFrame({
    'max_depth': results_dt_0['param_max_depth'],
    'min_samples_split': results_dt_0['param_min_samples_split'],
    'mean_test_score': results_dt_0['mean_test_score']})

max_depth_values_dt_0 = results_df_dt_0['max_depth']
mean_test_scores_dt_0 = results_df_dt_0['mean_test_score']
plt.plot(max_depth_values_dt_0, mean_test_scores_dt_0, marker='o', linestyle='-')
plt.title('Mean Test Scores for Different max_depth Values (Decision Tree)')
plt.xlabel('max_depth')
plt.ylabel('Mean Test Score')
plt.grid(True)
plt.show()
print(results_df_dt_0)

param_grid_dt = {'criterion': ['gini', 'entropy', 'log_loss'],
              'max_depth': np.arange(5, 10),                    #Tham số điền vào khi chạy ra bảng Different max_depth Values
              'min_samples_split': [2],                         #Tham số điền vào khi chạy ra bảng Different max_depth Values
              'min_samples_leaf': np.arange(1, 10)}
dt_model = DecisionTreeClassifier()
dt_grid = GridSearchCV(dt_model, param_grid_dt, cv=5)
dt_grid.fit(X_train, y_train)
best_model_dt = dt_grid.best_estimator_

y_pred_train_dt = best_model_dt.predict(X_train)
print("\n\t  Classification report for training set (Decision Tree)")
print("-"*55)
print(classification_report(y_train, y_pred_train_dt))

cm_dt = confusion_matrix(y_train, y_pred_train_dt)
plt.figure(figsize=(10, 5))
sns.heatmap(cm_dt, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix for Train Data - Decision Tree')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

y_pred_dt = best_model_dt.predict(X_test)
print("Accuracy (Decision Tree):", accuracy_score(y_test, y_pred_dt))
print("Classification Report (Decision Tree):")
print(classification_report(y_test, y_pred_dt))
print("Confusion Matrix (Decision Tree):")
print(confusion_matrix(y_test, y_pred_dt))

fig_dt, (ax1_dt, ax2_dt) = plt.subplots(1, 2, figsize=(15, 7), dpi=100)
ConfusionMatrixDisplay.from_estimator(best_model_dt, X_test, y_test, colorbar=False, ax=ax1_dt, cmap='Blues')
ax1_dt.set_title('Confusion Matrix for Test Data (Decision Tree)')
ax1_dt.grid(False)

RocCurveDisplay.from_estimator(best_model_dt, X_test, y_test, ax=ax2_dt)
ax2_dt.set_xlabel('False Positive Rate')
ax2_dt.set_ylabel('True Positive Rate')
ax2_dt.set_title('ROC Curve for Test Data (Decision Tree)')
plt.tight_layout()
plt.show()

y_scores_dt = best_model_dt.predict_proba(X_test)[:, 1]
print('ROC AUC Score (Decision Tree):', roc_auc_score(y_test, y_scores_dt))
print('Decision Tree Training Accuracy: {:.3f}'.format(best_model_dt.score(X_train, y_train)))
print('Decision Tree Test Accuracy: {:.3f}'.format(best_model_dt.score(X_test, y_test)))
plt.figure(figsize=(20,10))
tree.plot_tree(best_model_dt, filled=True, feature_names=X_test.columns, class_names=['No', 'Yes'])
plt.show()

#%%
### Random Forest

rf_model = RandomForestClassifier(random_state=42, max_depth=9, min_samples_leaf=4, min_samples_split=10, n_estimators=50)
rf_model.fit(X_train, y_train)

y_train_pred_rf = rf_model.predict(X_train)
y_test_pred_rf = rf_model.predict(X_test)

train_accuracy_rf = accuracy_score(y_train, y_train_pred_rf)
test_accuracy_rf = accuracy_score(y_test, y_test_pred_rf)

print("Accuracy on training set (Random Forest):", train_accuracy_rf)
print("Accuracy on test set (Random Forest):", test_accuracy_rf)

print("Confusion Matrix (Random Forest):")
print(confusion_matrix(y_test, y_test_pred_rf))
print("\nClassification Report (Random Forest):")
print(classification_report(y_test, y_test_pred_rf))

cm_rf = confusion_matrix(y_test, y_test_pred_rf)
cm_display_rf = ConfusionMatrixDisplay(confusion_matrix=cm_rf, display_labels=rf_model.classes_)
cm_display_rf.plot(cmap='Blues', values_format='d')
plt.title('Confusion Matrix (Random Forest)')
plt.show()
#%%
### Neural Network

param_dist_nn = {'hidden_layer_sizes': [(50, 50)],
                'activation': ['tanh'],
                'solver': ['adam'],
                'alpha': [0.001, 0.01, 0.1],
                'learning_rate': ['constant', 'adaptive']}
nn_model = MLPClassifier(random_state=42)

random_search_nn = RandomizedSearchCV(nn_model, param_distributions=param_dist_nn, n_iter=25, cv=5, scoring='accuracy', random_state=42)
random_search_nn.fit(X_train, y_train)

print("Best Parameters (Neural Network):")
print(random_search_nn.best_params_)

nn_model = MLPClassifier(random_state=42, activation='tanh', solver='adam', learning_rate='constant', hidden_layer_sizes=(50,50), alpha=0.1)
nn_model.fit(X_train, y_train)

y_train_pred_nn = nn_model.predict(X_train)
y_test_pred_nn = nn_model.predict(X_test)

train_accuracy_nn = accuracy_score(y_train, y_train_pred_nn)
test_accuracy_nn = accuracy_score(y_test, y_test_pred_nn)

print("Accuracy on training set (Neural Network):", train_accuracy_nn)
print("Accuracy on test set (Neural Network):", test_accuracy_nn)

print("Confusion Matrix (Neural Network):")
print(confusion_matrix(y_test, y_test_pred_nn))
print("\nClassification Report (Neural Network):")
print(classification_report(y_test, y_test_pred_nn))

cm_nn = confusion_matrix(y_test, y_test_pred_nn)
cm_display_nn = ConfusionMatrixDisplay(confusion_matrix=cm_nn, display_labels=nn_model.classes_)
cm_display_nn.plot(cmap='Blues', values_format='d')
plt.title('Confusion Matrix (Neural Network)')
plt.show()
#%%
### Support Vector Machine

class_weights_svm = [{0: x, 1: 1.0 - x} for x in np.linspace(0.001, 0.5, 12)]
param_dist_svm = {'kernel': ['rbf', 'sigmoid', 'polynomial'],
              'gamma': [1, 0.1, 0.01, 0.001],
              'C': [0.01, 0.1, 1, 10, 100],
              'class_weight': class_weights_svm}

svm_model = SVC(probability=True, decision_function_shape='ovr')
cv_svm = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
clf_random_svm = RandomizedSearchCV(svm_model, param_dist_svm, n_iter=15, cv=cv_svm, scoring='accuracy', random_state=42, n_jobs=-1)
clf_random_svm.fit(X_train, y_train)

print("Best Parameters (SVM):", clf_random_svm.best_params_)
print("Best Accuracy (SVM):", clf_random_svm.best_score_)

best_model_svm = clf_random_svm.best_estimator_

y_pred_train_svm = best_model_svm.predict(X_train)
print("\n\t  Classification report for training set (SVM)")
print("-"*55)
print(classification_report(y_train, y_pred_train_svm))

fig_svm, (ax1_svm, ax2_svm) = plt.subplots(1, 2, figsize=(15, 7), dpi=100)
ConfusionMatrixDisplay.from_estimator(best_model_svm, X_train, y_train, colorbar=False, ax=ax1_svm)
ax1_svm.set_title('Confusion Matrix for Training Data (SVM)')
ax1_svm.grid(False)

RocCurveDisplay.from_estimator(best_model_svm, X_train, y_train, ax=ax2_svm)
ax2_svm.set_xlabel('False Positive Rate')
ax2_svm.set_ylabel('True Positive Rate')
ax2_svm.set_title('ROC Curve for Training Data (SVM)')
plt.tight_layout()
plt.show()

y_pred_test_svm = best_model_svm.predict(X_test)
print("\n\t   Classification report for Test set (SVM)")
print("-"*55)
print(classification_report(y_test, y_pred_test_svm))

fig, (ax1_svm, ax2_svm) = plt.subplots(1, 2, figsize=(15, 7), dpi=100)
ConfusionMatrixDisplay.from_estimator(best_model_svm, X_test, y_test, colorbar=False, ax=ax1_svm)
ax1_svm.set_title('Confusion Matrix for Test Data (SVM)')
ax1_svm.grid(False)

RocCurveDisplay.from_estimator(best_model_svm, X_test, y_test, ax=ax2_svm)
ax2_svm.set_xlabel('False Positive Rate')
ax2_svm.set_ylabel('True Positive Rate')
ax2_svm.set_title('ROC Curve for Test Data (SVM)')
plt.tight_layout()
plt.show()

train_accuracy_svm = best_model_svm.score(X_train, y_train)
test_accuracy_svm = best_model_svm.score(X_test, y_test)
print("Accuracy on Training Data (SVM):", train_accuracy_svm)
print("Accuracy on Test Data (SVM):", test_accuracy_svm)
#%% - In ra các kết quả của các mô hình:
def metrics_calculator(clf, X_test, y_test, model_name):
    y_pred = clf.predict(X_test)
    result = pd.DataFrame(data=[accuracy_score(y_test, y_pred),
                                precision_score(y_test, y_pred, average='binary'),
                                recall_score(y_test, y_pred, average='binary'),
                                f1_score(y_test, y_pred, average='binary')],
                          index=['Accuracy', 'Precision', 'Recall', 'F1-score'],
                          columns=[model_name])
    result = (result * 100).round(2).astype(str) + '%'
    return result
# Save the final performance classifier
dt_result = metrics_calculator(best_model_dt, X_test, y_test, 'Decision Tree')
rf_result = metrics_calculator(rf_model, X_test, y_test, 'Random Forest')
nn_result = metrics_calculator(nn_model, X_test, y_test, 'Neural Network')
svm_result = metrics_calculator(best_model_svm, X_test, y_test, 'Support Vector Machine')

# Performance results into a single dataframe
results = pd.concat([svm_result, dt_result,rf_result, nn_result], axis=1).T
results.sort_values(by='F1-score', ascending=False, inplace=True)
