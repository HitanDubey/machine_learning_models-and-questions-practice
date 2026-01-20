import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline

df = pd.read_csv(r'ml\naive_Bayes\spam.csv')
# print(df.head())
print(df.groupby('Category').describe())
df['spam'] = df['Category'].apply(lambda x : 1 if x=='spam' else 0)
print(df.head())
x_train,x_test,y_train,y_test = train_test_split(df.Message,df.spam,test_size=0.25)
print(len(x_train)," ",len(x_test))

#count vectorized technique

v = CountVectorizer()
X_train_count = v.fit_transform(x_train.values)
X_train_count.toarray()[:2]
from sklearn.naive_bayes import MultinomialNB
model = MultinomialNB()
model.fit(X_train_count,y_train)

emails = [
    'Hey mohan, can we get together to watch footbal game tomorrow?',
    'Upto 20% discount on parking, exclusive offer just for you. Dont miss this reward!'
]
emails_count = v.transform(emails)
print(model.predict(emails_count))


X_test_count = v.transform(x_test)
print(model.score(X_test_count, y_test))
print(model.score(X_train_count,y_train))