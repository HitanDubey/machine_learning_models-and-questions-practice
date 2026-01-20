import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris

iris = load_iris()
print(dir(iris))
df = pd.DataFrame(iris.data,columns=iris.feature_names)
print(df.head(2))
df['target'] = iris.target 
# df['target_names'] = iris.target_names
print(df.head(2))
x = df.drop('target',axis='columns')
y =  df.target
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)
model = LogisticRegression(max_iter=1000)
model.fit(x_train,y_train)
print(model.score(x_test,y_test))
print(model.predict(x_test))