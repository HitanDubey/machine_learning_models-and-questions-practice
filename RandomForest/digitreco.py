import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits


digits = load_digits()
df = pd.DataFrame(digits.data)
print(df.head())
df['target'] = digits.target
print(df.head())
x = df.drop('target',axis='columns')
y = df.target

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=12)
model = RandomForestClassifier(n_estimators=12)
model.fit(x_train,y_train)
print(model.score(x_test,y_test))
print(model.score(x_train,y_train))