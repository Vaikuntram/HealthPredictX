from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import pickle

print('Heart Disease Dataset\n')

data = pd.read_csv('Datasets/heart_failure_clinical_records_dataset.csv')

#Data Cleaning and Preprocessing

print(data.isnull().sum())
data = data.fillna(data.median())

X = data.drop('DEATH_EVENT', axis=1)
y = data['DEATH_EVENT']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

#Model Training

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

'''
print('Random Forest Classifier\n')

rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train, y_train)
y_pred = rf_classifier.predict(X_test)

# Calculate and display the accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Random Forest Accuracy: {accuracy:.2f}")

'''

print('Logistic Regression\n')

log_reg = LogisticRegression(max_iter=1000)

param_grid = {
    'penalty': ['l1', 'l2', 'elasticnet', 'none'],
    'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
    'solver': ['liblinear', 'saga', 'lbfgs'],
    'l1_ratio': np.linspace(0, 1, 10)  # For elasticnet
}

grid_search = GridSearchCV(estimator=log_reg, param_grid=param_grid, cv=5, n_jobs=1, verbose=2)
grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_
y_pred_best = best_model.predict(X_test)
accuracy_best_model = accuracy_score(y_test, y_pred_best)
print(f"GridSearchCV Best Model Accuracy: {accuracy_best_model:.2f}")

log_reg.fit(X_train, y_train)
y_pred_log_reg = log_reg.predict(X_test)
accuracy_log_reg = accuracy_score(y_test, y_pred_log_reg)
print(f"Logistic Regression Accuracy: {accuracy_log_reg:.2f}")

'''
with open('logistic_model_updated.sav','wb') as file:
    pickle.dump(best_model, file)
'''
'''
print('SVC\n')

svc = SVC()
svc.fit(X_train, y_train)

y_pred_svc = svc.predict(X_test)
accuracy_svc = accuracy_score(y_test, y_pred_svc)
print(f"SVC Accuracy: {accuracy_svc:.2f}")

print('Decision Tree Classifier\n')

decision_tree = DecisionTreeClassifier(random_state=42)
decision_tree.fit(X_train, y_train)
y_pred_decision_tree = decision_tree.predict(X_test)
accuracy_decision_tree = accuracy_score(y_test, y_pred_decision_tree)
print(f"Decision Tree Accuracy: {accuracy_decision_tree:.2f}")

print('Gradient Boosting Classifier\n')

gradient_boosting = GradientBoostingClassifier(random_state=42)
gradient_boosting.fit(X_train, y_train)
y_pred_gradient_boosting = gradient_boosting.predict(X_test)
accuracy_gradient_boosting = accuracy_score(y_test, y_pred_gradient_boosting)
print(f"Gradient Boosting Accuracy: {accuracy_gradient_boosting:.2f}")

'''


