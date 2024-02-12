import warnings
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier , GradientBoostingClassifier , VotingClassifier
from sklearn.model_selection import GridSearchCV, cross_validate, RandomizedSearchCV, validation_curve

from xgboost import XGBClassifier
from catboost import  CatBoostClassifier
from lightgbm import LGBMClassifier

# !pip install catboost
# !pip install xgboost
# !pip install lightgbm

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)

warnings.simplefilter(action='ignore', category=Warning)

df=pd.read_csv("Dataset/diabetes.csv")

y=df["Outcome"]
X=df.drop(["Outcome"],axis=1)

####### RANDOM FOREST #########

rf_model = RandomForestClassifier()
rf_model.get_params()

cv_result = cross_validate(rf_model, X, y , cv=10, scoring=["accuracy","f1","roc_auc"])
cv_result["test_accuracy"].mean()
cv_result["test_f1"].mean()
cv_result["test_roc_auc"].mean()

