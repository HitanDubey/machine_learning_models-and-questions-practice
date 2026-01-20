import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
df = pd.read_csv(r'ml\logisticregression.py\insurance_data.csv')
print(df.head(2))
plt.xlabel('age')
plt.ylabel('insurance')
plt.scatter(df.age,df.insurance,marker="+",color='red')
plt.show()                                                     #sigmoid or logistic function
x_train,x_test,y_train,y_test = train_test_split(df[['age']],df.insurance,test_size=0.1,random_state=20)#      df[['age]]
model = LogisticRegression()
model.fit(x_train,y_train)
print(model.predict(pd.DataFrame([[12]],columns=['age'])))
print(model.score(x_test,y_test))