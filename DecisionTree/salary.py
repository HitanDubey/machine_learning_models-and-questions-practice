import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn import tree
df = pd.read_csv(r'ml\DecisionTree\salaries.csv')
input = df.drop('salary_more',axis='columns')
target = df['salary_more']
print(input)
le_company = LabelEncoder()
le_job = LabelEncoder()
le_degree = LabelEncoder()
input['company_n'] = le_company.fit_transform(input['company'])
input['job_n'] = le_job.fit_transform(input['job'])
input['degree_n'] = le_degree.fit_transform(input['degree'])
# print(input)
x = input.drop(['company','job','degree'],axis='columns')
print(x)
print(target)
model = tree.DecisionTreeClassifier()
x_train,x_test,y_train,y_test = train_test_split(x,target,test_size=0.2,random_state=20)
model.fit(x_train,y_train)
print(model.score(x_test,y_test))
print(model.predict(pd.DataFrame([['1','1','1']],columns=['company_n','job_n','degree_n'])))
print(model.predict(x_test))