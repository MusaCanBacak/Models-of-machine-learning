import pandas as pd
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)

df=pd.read_csv("Dataset/diabetes.csv")

y=df["Outcome"]
X=df.drop(["Outcome"],axis=1)

X_scaled=StandardScaler().fit_transform(X)
X=pd.DataFrame(X_scaled,columns=X.columns)

knn_model=KNeighborsClassifier().fit(X,y)
random_user=X.sample(1)
knn_model.predict(random_user)

y_pred=knn_model.predict(X)
y_prob=knn_model.predict_proba(X)[:,1]

print(classification_report(y,y_pred))
roc_auc_score(y,y_prob)

cv=cross_validate(knn_model,X,y,cv=10,scoring=["f1","roc_auc","accuracy"])

cv["test_f1"].mean()
cv["test_accuracy"].mean()
cv["test_roc_auc"].mean()


knn_model=KNeighborsClassifier()
knn_model.get_params()

knn_params={"n_neighbors":range(2,50)}

knn_best_p=GridSearchCV(knn_model,knn_params,cv=10,n_jobs=-1,verbose=1).fit(X,y)
knn_best_p.best_params_

knn_final=knn_model.set_params(**knn_best_p.best_params_).fit(X,y)

cv_result=cross_validate(X,y,cv=10,scoring=["f1","accuracy","roc_auc"])

cv_result["test_f1"].mean()
cv_result["test_accuracy"].mean()
cv_result["test_roc_auc"].mean()
