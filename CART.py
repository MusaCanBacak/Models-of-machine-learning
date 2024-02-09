import warnings
import joblib
import pydotplus
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.tree import DecisionTreeClassifier, export_graphviz, export_text
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split, GridSearchCV, cross_validate, validation_curve
from skompiler import skompile
import graphviz

# pip install pydotplus
# pip install skompiler
# pip install astor
# pip install joblib
# pip install garphviz

df = pd.read_csv("Dataset/diabetes.csv")

y = df["Outcome"]
X = df.drop(["Outcome"],axis=1)

cart_model=DecisionTreeClassifier().fit(X,y)

y_pred=cart_model.predict(X)
y_prob=cart_model.predict_proba(X)[:,1]

print(classification_report(y,y_pred))

roc_auc_score(y,y_prob)

X_train, y_train, X_test, y_test = train_test_split(X, y, test_size=0.20)
cart_model = DecisionTreeClassifier().fit(X_train ,y_train)

y_pred = cart_model.predict(X_train)
y_prob = cart_model.predict_proba(X_train)[:,1]
print(classification_report(y_test,y_pred))
roc_auc_score(y_test,y_prob)

cart_model = DecisionTreeClassifier().fit(X,y)
cv_result = cross_validate(cart_model, X, y, cv=10, scoring=["accuracy","f1","roc_auc"])

cv_result["test_accuracy"].mean()
cv_result["test_f1"].mean()
cv_result["test_roc_auc"].mean


