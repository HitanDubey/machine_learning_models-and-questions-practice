import pandas as pd
from sklearn.cluster import KMeans
import numpy as np
from matplotlib import pyplot as plt


df = pd.read_csv(r'ml\k_means_cluster\income.csv')
print(df.isnull().sum())
sse = []
k_rng = range(1,10)
for k in k_rng:
    km = KMeans(n_clusters=k)
    km.fit(df[['age','income']])
    sse.append(km.inertia_)

plt.xlabel('K')
plt.ylabel('Sum of squared error')
plt.plot(k_rng,sse)
plt.show()