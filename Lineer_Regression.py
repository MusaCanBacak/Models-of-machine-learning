import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option('display.float_format', lambda x: '%.2f' % x)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,mean_absolute_error
from sklearn.model_selection import train_test_split,cross_val_score

df=pd.read_csv("Dataset/advertising.csv") # dataset to use

X=df[["TV"]] #dependent variable
y=df[["sales"]] # independent variable

reg_model=LinearRegression().fit(X,y) # creating model

bias=reg_model.intercept_[0] # finding bias
weights=reg_model.coef_[0][0] # finding wights

# question: How much sales are expected if 150 units of TV are spent?
answer=bias+weights*150

# question2: How much sales are expected if 500 units of TV are spent?
answer2=bias+weights*500


