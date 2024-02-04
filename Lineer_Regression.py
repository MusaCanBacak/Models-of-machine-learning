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

graphic = sns.regplot(x=X, y=y, scatter_kws={'color': 'b', 's': 9},ci=False, color="r")

graphic.set_title(f"Model equation : Sales = {round(bias, 2)} + TV*{round(weights, 2)}")
graphic.set_ylabel("number of sales")
graphic.set_xlabel("TV Expenses")
plt.xlim(-10, 310)
plt.ylim(bottom=0)
plt.show()

# Evaluation of the model
y_pred = reg_model.predict(X)

mean_squared_error(y,y_pred) # MSE

mean_absolute_error(y,y_pred) # MAE

np.sqrt(mean_squared_error(y,y_pred)) # RMSE

reg_model.score(X,y) # R-squared

