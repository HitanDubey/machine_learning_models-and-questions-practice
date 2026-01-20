import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_digits

digits = load_digits()
print(dir(digits))
df = pd.DataFrame(digits.data)
print(df.head())
df['target'] = digits.target
print(df.head())
x = df.drop('target',axis='columns')
y = df.target
# X_train, X_test, y_train, y_test = train_test_split(x,y,test_size=0.3)
# model = SVC()
# model.fit(X_train,y_train)
# print(model.score(X_test,y_test))
# print(model.predict())
print(cross_val_score(LogisticRegression(max_iter=1000),x,y,cv=3))
print(cross_val_score(SVC(),x,y,cv=3))
print(cross_val_score(RandomForestClassifier(n_estimators=50),x,y,cv=3))
print(cross_val_score(DecisionTreeClassifier(),x,y,cv=3))
