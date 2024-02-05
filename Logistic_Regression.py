import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,roc_auc_score,confusion_matrix,classification_report,plot_roc_curve
from sklearn.model_selection import train_test_split,cross_val_score

def plot_numerical_col(dataframe, numerical_col):   #graphing function
    dataframe[numerical_col].hist(bins=20)
    plt.xlabel(numerical_col)
    plt.show(block=True)

def target_summary_with_num(dataframe, target, numerical_col):
    print(dataframe.groupby(target).agg({numerical_col: "mean"}), end="\n\n\n")

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