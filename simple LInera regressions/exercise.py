import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
df = pd.read_csv(r'ml\canada_per_capita_income.csv')
print(df.head(3))
plt.xlabel('year')
plt.ylabel('income')
plt.scatter(df.year,df.income,marker='+')
new_df = df.drop('income',axis='columns')
model = LinearRegression()
model.fit(new_df,df.income)
print(model.predict(pd.DataFrame([[2016]],columns=['year'])))
print(model.score(new_df,df.income))