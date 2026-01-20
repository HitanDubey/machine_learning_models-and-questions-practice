import pandas as pd
from sklearn import linear_model
from word2number import w2n
df = pd.read_csv(r'ml\multilinear\hiring.csv')
df.experience = df.experience.apply(w2n.word_to_num)
print(df.head(2))
model = linear_model.LinearRegression()
new_df = df.drop('salary',axis='columns')
model.fit(new_df,df.salary)
print(model.predict(pd.DataFrame([[2,8,6]],columns=['experience','test','interview'])))
print(model.score(new_df,df.salary))