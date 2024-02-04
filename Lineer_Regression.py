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