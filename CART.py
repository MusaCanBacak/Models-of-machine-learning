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
# conda install graphviz

df = pd.read_csv("Dataset/diabetes.csv")

y = df["Outcome"]
X = df.drop(["Outcome"],axis=1)

cart_model = DecisionTreeClassifier().fit(X,y)

y_pred = cart_model.predict(X)
y_prob = cart_model.predict_proba(X)[:, 1]

print(classification_report(y, y_pred))

roc_auc_score(y, y_prob)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)
cart_model = DecisionTreeClassifier().fit(X_train ,y_train)

y_pred = cart_model.predict(X_train)
y_prob = cart_model.predict_proba(X_train)[:, 1]
print(classification_report(y_train,y_pred))
roc_auc_score(y_train, y_prob)

# Cross validation
cart_model = DecisionTreeClassifier().fit(X,y)
cv_result = cross_validate(cart_model, X, y, cv=10, scoring=["accuracy","f1","roc_auc"])

cv_result["test_accuracy"].mean()
cv_result["test_f1"].mean()
cv_result["test_roc_auc"].mean

#Hyperparameter optimization
cart_model.get_params()

cart_params = {'max_depth': range(1, 11),
               "min_samples_split": range(2, 20)}

cart_best_grid = GridSearchCV(cart_model,
                              cart_params,
                              cv=10,
                              n_jobs=-1,
                              verbose=1).fit(X, y)
cart_best_grid.best_params_

cart_best_grid.best_score_

#Final model

cart_fnal=DecisionTreeClassifier(**cart_best_grid.best_params_).fit(X,y)
cart_fnal.get_params()

cart_final = cart_model.set_params(**cart_best_grid.best_params_).fit(X, y)

cv_result=cross_validate(cart_fnal, X, y, cv=10 , scoring=["accuracy","f1","roc_auc"])

cv_result['test_accuracy'].mean()
cv_result["test_f1"].mean()
cv_result["test_roc_auc"].mean()

def plot_importance(model, features, num=len(X), save=False): #feature importance graph
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

plot_importance(cart_fnal, X, num=5)

# def tree_graph(model, col_name, file_name):
#    tree_str = export_graphviz(model,feature_names=col_name,filled=True, out_file=None)
#    graph = pydotplus.graph_from_dot_data(tree_str)
#    graph.write_png(file_name)

#tree_graph(model=cart_final, col_name=X.columns, file_name="Cart_final.png")
#cart_final.get_params()

tree_rules = export_text(cart_final, feature_names=list(X.columns))
print(tree_rules)

#saveing the model
joblib.dump(cart_final,"cart_final.pkl")
# loading the model
cart_model_from_disc = joblib.load("cart_final.pkl")