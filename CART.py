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