import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

# Unsupervised learning (gözetimsiz öğrenme), etiketli verilerin olmadığı bir öğrenme türüdür. Bu öğrenme yaklaşımında, model verilerdeki gizli yapıları, desenleri veya grupları bulmaya çalışır. Model verilerde bir etiket veya hedef olmadan, sadece verilerin dağılımına bakarak bazı çıkarımlar yapar.

# Unsupervised Learning Türleri:
# Kümeleme (Clustering): Verileri benzerliklerine göre gruplamayı amaçlar. En popüler algoritmalarından biri K-Means'tir.
# Boyut indirgeme (Dimensionality Reduction): Yüksek boyutlu verilerdeki ana bileşenleri bulmayı hedefler. Bu alanda sık kullanılan teknikler arasında PCA (Principal Component Analysis) bulunur.
# Anomali tespiti (Anomaly Detection): Normal veri örneklerinden sapma gösteren anormal verileri tespit eder.
# Örnek: K-Means ile Kümeleme
# K-Means algoritması, verileri belirli sayıda küme içerisine bölerek her kümenin merkezini belirlemeye çalışır. Her veri noktası, bu merkezlere en yakın olan kümeye atanır.

# creating example set 
X, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)

# K-Means model
kmeans = KMeans(n_clusters=4)
kmeans.fit(X)

# to find which models with predict
y_kmeans = kmeans.predict(X)

# get center of cluster
centers = kmeans.cluster_centers_

# make visualize
plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50, cmap='viridis')
plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=0.75, marker='X')
plt.title('K-Means results')
plt.show()
