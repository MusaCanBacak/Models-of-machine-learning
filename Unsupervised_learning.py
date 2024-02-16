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
from sklearn.cluster import AgglomerativeClustering

df = pd.read_csv("Dataset/USArrests.csv", index_col=0)
print(df)

df.head()
df.isnull().sum()
df.info
df.describe().T

sc = MinMaxScaler((0, 1))
df = sc.fit_transform(df)
df[0:5]

kmeans = KMeans(n_clusters=4).fit(df)
kmeans.get_params()

kmeans.n_clusters
kmeans.cluster_centers_
kmeans.labels_
kmeans.inertia_

kmeans = KMeans()
ssd = []
K = range(1, 30)

for k in K:
    kmeans = KMeans(n_clusters=k).fit(df)
    ssd.append(kmeans.inertia_)

plt.plot(K, ssd, 'bx-')
plt.xlabel('Küme Sayısı')
plt.ylabel('Toplam Hata Kareleri')
plt.title('Küme Sayısı ile Toplam Hata Kareleri Arasındaki İlişki')
plt.show()

kmeans = KMeans()
elbow = KElbowVisualizer(kmeans, k=(1, 30))
elbow.fit(df)
elbow.show()

elbow.elbow_value_

kmeans = KMeans(n_clusters=elbow.elbow_value_)
cluster_kmeans = kmeans.labels_

df["cluster"] = cluster_kmeans
df.head()

df["cluster"] = df["cluster"] + 1
df[df["cluster"]==5]

df.groupby("cluster").agg(["count","mean","median"])

df.to_csv("clusters.csv")

############## Hierarchical Clustering ##############

df = pd.read_csv("Dataset/USArrests.csv", index_col=0)
sc = MinMaxScaler((0, 1))
df = sc.fit_transform(df)

hc_average = linkage(df, "average")

plt.figure(figsize=(14, 7))
plt.title("Hierarchical Clustering Dendrogram")
plt.xlabel("Observation Units")
plt.ylabel("Distances")
dendrogram(hc_average, leaf_font_size=10)
plt.show()

plt.figure(figsize=(7, 5))
plt.title("Dendrograms")
dend = dendrogram(hc_average)
plt.axhline(y=0.5, color='r', linestyle='--')
plt.axhline(y=0.6, color='b', linestyle='--')
plt.show()

cluster = AgglomerativeClustering(n_clusters = 5, linkage= "average")

cluster = cluster.fit_predict(df)

df["hi_cluster_no"] = cluster
df["hi_cluster_no"] = df["hi_cluster_no"] +1

df["kmeans_cluster_no"] = df["kmeans_cluster_no"] + 1
df["kmeans_cluster_no"] = cluster_kmeans


