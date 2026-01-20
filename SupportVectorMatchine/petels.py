import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.svm import SVC

iris = load_iris()
# print(iris.feature_names)
# print(iris.target_names)
df = pd.DataFrame(iris.data,columns=iris.feature_names)
print(df.head())
df['target'] = iris.target
# print(df.head())
print(df[df.target==1].head())
print(df[df.target==2].head())
df['flower_name'] =df.target.apply(lambda x: iris.target_names[x])
df0 = df[:50]
df1 = df[50:100]
df2 = df[100:]

plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.scatter(df0['sepal length (cm)'], df0['sepal width (cm)'],color="green",marker='+')
plt.scatter(df1['sepal length (cm)'], df1['sepal width (cm)'],color="red",marker='.')
plt.show()


plt.xlabel('Petal Length')
plt.ylabel('Petal Width')
plt.scatter(df0['petal length (cm)'], df0['petal width (cm)'],color="black",marker='+')
plt.scatter(df1['petal length (cm)'], df1['petal width (cm)'],color="red",marker='.')
plt.show()

X = df.drop(['target','flower_name'], axis='columns')
y = df.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = SVC()
model.fit(X_train, y_train)
print("Model Score : ",model.score(X_test, y_test))
print("Predicted value of : [[4.8,3.0,1.5,0.3]] = ", model.predict(pd.DataFrame([[4.8,3.0,1.5,0.3]],columns=['sepal length (cm)','sepal width (cm)','petal length (cm)','petal width (cm)'])))