import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
digit = load_digits()
# print(digit)
# print(dir(digit))
# print(digit.images[0])
# plt.gray()
# plt.matshow(digit.images[9])
# plt.show()
x_train,x_test,y_train,y_test = train_test_split(digit.data,digit.target,test_size=0.2,random_state=42)
model = LogisticRegression(max_iter=1000)
model.fit(x_train,y_train)
print(model.score(x_test,y_test))
y_predicted = model.predict(x_test)
cm = confusion_matrix(y_test,y_predicted)
print(cm)



import seaborn as sn
plt.figure(figsize = (10,7))
sn.heatmap(cm, annot=True)
plt.xlabel('Predicted')
plt.ylabel('Truth')
plt.show()