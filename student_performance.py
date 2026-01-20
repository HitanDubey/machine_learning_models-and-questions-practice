import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score,confusion_matrix
import seaborn as sns

df = pd.read_csv(r'ml\StudentPerformance.csv')
print(df.head())
plt.scatter(df.Hours_Studied,df.Performance_Index,marker='+',color='blue')
plt.show()
plt.scatter(df.Previous_Scores,df.Performance_Index,marker='o',color='red')
plt.show()
df.Extracurricular_Activities = df.Extracurricular_Activities.map({'Yes':1,'No':0})
x = df.drop('Performance_Index',axis='columns')
y = df.Performance_Index
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
model = LinearRegression()
model.fit(x_train,y_train)
print(round(model.score(x_test,y_test)*100,2),"%")
print(round(model.score(x_train,y_train)*100,2),"%")
cm = confusion_matrix(y_test, model.predict(x_test).round())
print(cm)
