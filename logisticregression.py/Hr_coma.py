import pandas as pd
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

df = pd.read_csv(r'ml\logisticregression.py\HR_comma_sep.csv')
left = df[df.left==1]
left.shape
retained = df[df.left==0]
retained.shape
pd.crosstab(df.salary,df.left).plot(kind='bar')
plt.show()
pd.crosstab(df.Department,df.left).plot(kind='bar')
plt.show()
subdf = df[['satisfaction_level','average_montly_hours','promotion_last_5years','salary']]
# print(subdf.head())
salary_dummies = pd.get_dummies(subdf.salary, prefix="salary")
df_with_dummies = pd.concat([subdf,salary_dummies],axis='columns')
# df_with_dummies.head()
df_with_dummies.drop('salary',axis='columns',inplace=True)
# print(df_with_dummies.head())
X = df_with_dummies
y = df.left
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3)
model = LogisticRegression(max_iter=1000)
model.fit(X_train,y_train)
print(model.score(X_test,y_test))
print(model.predict(X_test))
