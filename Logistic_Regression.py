import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,roc_auc_score,confusion_matrix,classification_report,plot_roc_curve
from sklearn.model_selection import train_test_split,cross_val_score

def plot_numerical_col(dataframe, numerical_col):  #graphing function
    dataframe[numerical_col].hist(bins=20)
    plt.xlabel(numerical_col)
    plt.show(block=True)

def target_summary_with_num(dataframe, target, numerical_col):
    print(dataframe.groupby(target).agg({numerical_col: "mean"}), end="\n\n\n")

def outlier_thresholds(dataframe, col_name, q1=0.25, q3=0.75): #finding upper and lower limit
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False

def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

def plot_confusion_matrix(y, y_pred):
    acc = round(accuracy_score(y, y_pred), 2)
    cm = confusion_matrix(y, y_pred)
    sns.heatmap(cm, annot=True, fmt=".0f")
    plt.xlabel('y_pred')
    plt.ylabel('y')
    plt.title('Accuracy Score: {0}'.format(acc), size=10)
    plt.show()

df=pd.read_csv("Dataset/diabetes.csv") #import dataset

df["Outcome"].value_counts()    #independent variable analysis
sns.countplot(x="Outcome",data=df)
plt.show()

100 * df["Outcome"].value_counts() / len(df)

for col in df.columns:
    plot_numerical_col(df,col)

dependent_cols=[col for col in df.columns if "Outcome" not in col]

for col in dependent_cols:
    target_summary_with_num(df, "Outcome", col)

for col in dependent_cols:
    print(col, check_outlier(df, col))

replace_with_thresholds(df, "Insulin")

for col in dependent_cols:
    df[col] = RobustScaler().fit_transform(df[[col]])

df.head()

y=df["Outcome"]
X=df.drop(["Outcome"],axis=1)

logistic_model=LogisticRegression().fit(X,y) #train the model

y_pred=logistic_model.predict(X)

plot_confusion_matrix(y, y_pred)

print(classification_report(y, y_pred))

y_prob = logistic_model.predict_proba(X)[:, 1]
roc_auc_score(y, y_prob)

