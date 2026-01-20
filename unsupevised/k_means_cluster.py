import pandas as pd
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt
# import matplotlib.pyplot as plt

df = pd.read_csv(r'ml\unsupevised\income.csv')
plt.scatter(df.age,df['income'])
# plt.show()
km = KMeans(n_clusters=3)
y = km.fit_predict(df[['age','income']])
print(y)
df['cluster'] = y
# print(df)
df1 = df[df.cluster == 0]
df2 = df[df.cluster == 1]
df3 = df[df.cluster == 2]
# print(df1)    
# print(km.cluster_centers_)
plt.scatter(df1.age,df1['income'],color='green')
plt.scatter(df2.age,df2['income'],color='red')
plt.scatter(df3.age,df3['income'],color='blue')
plt.xlabel('Age')
plt.ylabel('Income ($)')
# plt.show()

scaler = MinMaxScaler()
scaler.fit(df[['income']])
df['income'] = scaler.transform(df[['income']])

scaler.fit(df[['age']])
df['age'] = scaler.transform(df[['age']])