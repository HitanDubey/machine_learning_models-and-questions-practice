import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
df = pd.read_csv(r'ml\onehotencoding\homeprices.csv')
le = LabelEncoder()
dfle = df
x = dfle[['town','area']].values
y = dfle.price.values
ohe = OneHotEncoder(categorical_features=[0])
ohe.fit_transform(x).toarray()
x = x[:,1:]
model = LinearRegression()
model.fit(x,y)
print(model.score(x,y))