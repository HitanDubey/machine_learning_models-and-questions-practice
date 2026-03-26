from sklearn.linear_model import LinearRegression
import pandas as pd
import matplotlib.pyplot as plt
df =  pd.read_csv(r"ml/ml.csv")
new_df = df.drop('price', axis='columns')
model = LinearRegression()
model.fit(new_df,df.price)
print(model.predict(pd.DataFrame([[5500]], columns=['area'])))
print(model.score(new_df, df['price']))
print(model.coef_)
print(model.intercept_)
areadf = pd.read_csv(r"ml/areas.csv")
p = model.predict(areadf)
areadf['price'] = p
print(areadf)
areadf.to_csv(r"ml/areas_with_prices.csv", index=False)
print(model.score(areadf[['area']], areadf['price'])) 