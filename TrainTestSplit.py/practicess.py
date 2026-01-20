import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

df = pd.read_csv(r'ml\TrainTestSplit.py\carprices.csv')
# print(df.head())
x = df[['Mileage','Age']] # 2D requre for x axis 
# ### we can also use this instead of upper code #     x = df.drop('Sell Price',axis='columns')
y = df['Sell Price']
# print(x,"\n",y)
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=18)
model = LinearRegression()
model.fit(x_train,y_train)
# print(x_test,"\n\n")
# print(y_test)
print(model.score(x_test,y_test))
print(model.predict(x_test))