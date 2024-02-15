import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from yellowbrick.cluster import KElbowVisualizer
from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import dendrogram
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("Dataset/USArrests.csv", index_col=0)

df.head()
df.isnull().sum()
df.info
df.describe().T

sc = MinMaxScaler((0, 1))
df = sc.fit_transform(df)
df[0:5]

kMeans = KMeans(n_clusters=4).fit(df)
kMeans.get_params()
kMeans.cluster_centers_
kMeans.labels_
kMeans.inertia_
