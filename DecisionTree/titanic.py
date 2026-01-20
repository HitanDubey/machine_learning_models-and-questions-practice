import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import tree

df = pd.read_csv(r'ml\DecisionTree\titanic.csv')
df.drop(['PassengerId','Name','SibSp','Parch','Ticket','Cabin','Embarked'],axis='columns',inplace=True)
# print(df.head(3))
input = df.drop('Survived',axis='columns')
target = df.Survived
input.Sex = input.Sex.map({'male': 1, 'female': 2})
print(input.Sex)
# print(input.Age[:10])
input.Age = input.Age.fillna(input.Age.mean())
# print(input.Age[:10])
print(input.head(3))
X_train, X_test, y_train, y_test = train_test_split(input,target,test_size=0.2)
model = tree.DecisionTreeClassifier()
model.fit(X_train,y_train)
print(model.score(X_test,y_test))
print(model.predict(X_test))