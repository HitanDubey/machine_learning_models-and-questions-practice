import pandas as pd
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from  sklearn.naive_bayes import GaussianNB

df = pd.read_csv(r'ml\naive_Bayes\titanic.csv')
print(df.head())
# print(df.size)
# print(df.shape)
# print(df.nunique())
df.drop(['PassengerId','Name','Ticket','Cabin','Embarked','SibSp','Parch'],axis='columns',inplace=True)
# print(df.head())
# print(df.isna().sum())
# print(df.head(25))
df['Age'] = df['Age'].fillna(df['Age'].mean())
# print(df.head(25))
# print(df.isna().sum())
# print(df.shape)
# print(df.head())
target = df.Survived
input = df.drop('Survived',axis='columns')
# print(target,input)
dummy = pd.get_dummies(input.Sex)
print(dummy.head())
input = pd.concat([input,dummy],axis='columns')
print(input.head())
input.drop('Sex',axis='columns',inplace=True)
print(input.head())
x_train,x_test,y_train,y_test = train_test_split(input,target,test_size=0.3,random_state=233)
model = GaussianNB()
model.fit(x_train,y_train)
print(model.score(x_test,y_test))
print(model.score(x_train,y_train))
print(model.predict(x_test[:10]))