import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
iris = load_iris()
print(dir(iris))
df = pd.DataFrame(iris.data, columns=iris.feature_names)
print(df.head())
df['target'] = iris.target
print(df.head())

x_train,x_test,y_train,y_test = train_test_split(df.drop('target',axis='columns'),df.target,test_size=0.3)
model = SVC()
model.fit(x_train,y_train)
print(model.score(x_test,y_test))
print(model.score(x_train,y_train))
model = RandomForestClassifier()
model.fit(x_train,y_train)
print(model.score(x_train,y_train))
print(model.score(x_test,y_test))
model = DecisionTreeClassifier()
model.fit(x_train,y_train)
print(model.score(x_test,y_test))
print(model.score(x_train,y_train))