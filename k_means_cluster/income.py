import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import  MinMaxScaler
from matplotlib import pyplot as plt 

df = pd.read_csv(r'ml\k_means_cluster\income.csv')
plt.scatter(df.age,df.income)
# plt.show()

km = KMeans(n_clusters=3)
y_predicted = km.fit_predict(df[['age','income']])
y_predicted
df['cluster']=y_predicted
print(df.head())
km.cluster_centers_
df1 = df[df.cluster==0]
df2 = df[df.cluster==1]
df3 = df[df.cluster==2]
print(df1.head())
plt.scatter(df1.age,df1['income'],color='green')
plt.scatter(df2.age,df2['income'],color='red')
plt.scatter(df3.age,df3['income'],color='black')
plt.scatter(km.cluster_centers_[:,0],km.cluster_centers_[:,1],color='purple',marker='*',label='centroid')
plt.xlabel('age')
plt.ylabel('income')
plt.legend()
plt.show()

scaler = MinMaxScaler()

scaler.fit(df[['income']])
df['income'] = scaler.transform(df[['income']])

scaler.fit(df[['age']])
df['age'] = scaler.transform(df[['age']])
print(df.head())
plt.scatter(df.age,df['income'])
plt.show()

km = KMeans(n_clusters=3)
y_predicted = km.fit_predict(df[['age','income']])
print(y_predicted)
df['cluster']=y_predicted
print(df.head())
km.cluster_centers_

df1 = df[df.cluster==0]
df2 = df[df.cluster==1]
df3 = df[df.cluster==2]
plt.scatter(df1.age,df1['income'],color='green')
plt.scatter(df2.age,df2['income'],color='red')
plt.scatter(df3.age,df3['income'],color='black')
plt.scatter(km.cluster_centers_[:,0],km.cluster_centers_[:,1],color='purple',marker='*',label='centroid')
plt.legend()
plt.show()