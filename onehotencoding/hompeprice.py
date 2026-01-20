import pandas as pd
from sklearn.linear_model import LinearRegression
df = pd.read_csv(r'ml\onehotencoding\homeprices.csv')
dumies = pd.get_dummies(df.town)
# print(dumies)
df_dumies = pd.concat([df,dumies],axis='columns')
# print(df_dumies)
df_dumies.drop(['town','west windsor'],axis='columns',inplace=True)
print(df_dumies)
x = df_dumies.drop('price',axis='columns')
y = df_dumies.price
model  = LinearRegression()
model.fit(x,y)
print(model.score(x,y))
print(model.predict(pd.DataFrame([[2600,True,False]],columns=['area','monroe township','robinsville'])))
