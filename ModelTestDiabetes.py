from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
import pickle

print('Diabetes Dataset\n')

data = pd.read_csv('Datasets/diabetes.csv')

#Data Cleaning and Preprocessing

# Check for missing values
print(data.isnull().sum())

# Replace zeros with NaN in certain columns where zeros are not realistic
columns_to_replace_zero = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']

for col in columns_to_replace_zero:
    data[col].replace(0, np.nan, inplace=True)

# Fill missing values with mean or median of the column
for col in columns_to_replace_zero:
    data[col].fillna(data[col].median(), inplace=True)

# Verify that there are no more missing values
print(data.isnull().sum())

#Model Training

X = data.drop('Outcome', axis=1)
y = data['Outcome']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


print('SVC\n')

svc = SVC(probability=True)
svc.fit(X_train, y_train)

y_pred_svc = svc.predict(X_test)
accuracy_svc = accuracy_score(y_test, y_pred_svc)
print(f"SVC Accuracy: {accuracy_svc:.2f}")

with open('svc_diabetes.sav','wb') as file:
    pickle.dump(svc, file)














