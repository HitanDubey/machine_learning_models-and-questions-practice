import pandas as pd
from sklearn.linear_model import LinearRegression
df = pd.read_csv(r'ml\onehotencoding\carprices.csv')
dummy = pd.get_dummies(df.Car_Model)
# print(dummy)
df_dummies = pd.concat([df,dummy],axis='columns')
# print(df_dummies)
df_dummies.drop(['Car_Model','Mercedez Benz C class'],axis='columns',inplace=True)
print(df_dummies)
x = df_dummies.drop('Sell',axis='columns')
y = df_dummies.Sell
model = LinearRegression()
model.fit(x,y)
print(model.score(x,y))
print(model.predict(pd.DataFrame([[82000,1,False,True]],columns=['Mileage','Age','Audi A5','BMW X5'])))