import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import warnings
import seaborn as sns

warnings.filterwarnings('ignore')
df = pd.read_csv(r'ml\L1_L2_regularization\Melbourne_housing_FULL.csv')
# print(df.head())
# print(df.nunique())
cols_to_use = ['Suburb', 'Rooms', 'Type', 'Method', 'SellerG', 'Regionname', 'Propertycount', 
               'Distance', 'CouncilArea', 'Bedroom2', 'Bathroom', 'Car', 'Landsize', 'BuildingArea', 'Price']
df = df[cols_to_use]
# print(df.head())
# print(df.shape)
# print(df.isnull().sum())
cols_to_fill_zero = ['Propertycount', 'Distance', 'Bedroom2', 'Bathroom', 'Car']
df[cols_to_fill_zero] = df[cols_to_fill_zero].fillna(0)
# print(df.isnull().sum())
df['Landsize'] = df['Landsize'].fillna(df.Landsize.mean())
df['BuildingArea'] = df['BuildingArea'].fillna(df.BuildingArea.mean())
# print(df.isnull().sum())
# print(df.shape)
df.dropna(inplace=True)
# print(df.shape)
df = pd.get_dummies(df, drop_first=True)
print(df.head())
x = df.drop('Price',axis=1)
y = df.Price
from sklearn.model_selection import train_test_split
train_X, test_X, train_y, test_y = train_test_split(x, y, test_size=0.3, random_state=2)
from sklearn.linear_model import LinearRegression
model = LinearRegression().fit(train_X,train_y)
print(model.score(test_X,test_y))
print(model.score(train_X,train_y))

#L1 regularization
from sklearn import linear_model
lasso_reg = linear_model.Lasso(alpha=50, max_iter=100, tol=0.1)
lasso_reg.fit(train_X, train_y)
print(lasso_reg.score(train_X,train_y))
print(lasso_reg.score(test_X,test_y))

#L2 Regularization
from sklearn.linear_model import Ridge
ridge_reg= Ridge(alpha=50, max_iter=100, tol=0.1)
ridge_reg.fit(train_X, train_y)
print(ridge_reg.score(test_X, test_y))
print(ridge_reg.score(train_X, train_y))
