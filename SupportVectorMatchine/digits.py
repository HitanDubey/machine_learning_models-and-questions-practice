import pandas as pd
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
digits = load_digits()
print(digits.target)
print(dir(digits))
df = pd.DataFrame(digits.data,columns=digits.feature_names)
print(df.head(5))

df['target'] = digits.target
print(df.head(5))


X_train, X_test, y_train, y_test = train_test_split(df.drop('target',axis='columns'), df.target, test_size=0.3,random_state=40)


rbf_model = SVC(kernel='rbf')

rbf_model.fit(X_train, y_train)
print(rbf_model.score(X_test,y_test))

linear_model = SVC(kernel='linear')
linear_model.fit(X_train,y_train)

print(linear_model.score(X_test,y_test))