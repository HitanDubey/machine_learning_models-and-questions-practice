from sklearn.linear_model import LinearRegression
import pandas as pd
import matplotlib.pyplot as plt
df =  pd.read_csv(r"ml/ml.csv")
print(df)
plt.xlabel('area')
plt.ylabel('price')
plt.scatter(df.area,df.price,marker="+",color='red')
plt.show()
new_df = df.drop('price', axis='columns') #we can also write df[['area']]
model = LinearRegression()
model.fit(new_df,df.price)  #we  can also write df[['area']]
print(model.predict(pd.DataFrame([[4001]],columns=['area'])))
print(model.score(new_df, df['price'])*100)
print(model.coef_)      
print(model.intercept_)