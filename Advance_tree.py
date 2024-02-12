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

rf_params = {"max_depth" : [5,10,None],
             "max_features" : [3,5,7,"auto"],
             "min_samples_split":[3,5,9,13,17,20],
             "n_estimators":[100,200,300,400,500]}

rf_best_params = GridSearchCV(rf_model, rf_params, cv=10, n_jobs=-1, verbose=True).fit(X,y)
rf_best_params.best_params_

rf_final = rf_model.set_params(**rf_best_params.best_params_).fit(X,y)

cv_result = cross_validate(rf_final, X, y, cv=10 , scoring=["accuracy", "f1", "roc_auc"])

cv_result["test_accuracy"].mean()
cv_result["test_f1"].mean()
cv_result["test_roc_auc"].mean()

def plot_importance(model, features, num=len(X), save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                     ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig('importances.png')

plot_importance(rf_final, X)

def val_curve_params(model, X, y, param_name, param_range, scoring="roc_auc", cv=10):
    train_score, test_score = validation_curve(
        model, X=X, y=y, param_name=param_name, param_range=param_range, scoring=scoring, cv=cv)

    mean_train_score = np.mean(train_score, axis=1)
    mean_test_score = np.mean(test_score, axis=1)

    plt.plot(param_range, mean_train_score,
             label="Training Score", color='b')

    plt.plot(param_range, mean_test_score,
             label="Validation Score", color='g')

    plt.title(f"Validation Curve for {type(model).__name__}")
    plt.xlabel(f"Number of {param_name}")
    plt.ylabel(f"{scoring}")
    plt.tight_layout()
    plt.legend(loc='best')
    plt.show(block=True)

val_curve_params(rf_final, X, y, "max_depth", range(1, 11), scoring="roc_auc")

################## *GBM* ####################

gbm_model = GradientBoostingClassifier()
gbm_model.get_params()

cv_result_gbm = cross_validate(gbm_model ,X ,y ,cv=10 ,scoring=["accuracy","f1","roc_auc"])

cv_result_gbm["test_accuracy"].mean()
cv_result_gbm["test_f1"].mean()
cv_result_gbm["test_roc_auc"].mean()

gbm_params = {"learning_rate": [0.01, 0.1],
              "max_depth":[1,3,5,7,9],
              "min_samples_split":[1,2,3,5,7,10,12],
              "n_estimators": [100, 500, 700, 1000],
              "subsample": [1, 0.5, 0.7]}

gbm_best_params = GridSearchCV(gbm_model ,gbm_params ,cv=10 ,n_jobs=-1 ,verbose=True).fit(X,y)
gbm_best_params.best_params_

gbm_final= gbm_model.set_params(**gbm_best_params.best_params_).fit(X,y)

cv_result_gbm = cross_validate(gbm_model ,X ,y ,cv=10 ,scoring=["accuracy","f1","roc_auc"])
cv_result_gbm["test_accuracy"].mean()
cv_result_gbm["test_f1"].mean()
cv_result_gbm["test_roc_auc"].mean()

plot_importance(gbm_final, X)
val_curve_params(gbm_final, X, y, "max_depth", range(1, 11), scoring="roc_auc")

################## *XGBoost* ####################
