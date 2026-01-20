from sklearn import linear_model
import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv(r'ml\multilinear\homeprices.csv')
print(df)
model = linear_model.LinearRegression()
new_df = df.drop('price',axis='columns')
model.fit(new_df,df.price)
print(model.score(new_df,df.price))
print(model.predict(pd.DataFrame([[4100,6,8]],columns=['area','bedrooms','age'])))