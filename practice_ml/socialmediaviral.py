import pandas as pd

df = pd.read_csv(f'practice_ml\social_media_viral_content_dataset.csv')
# print(df.head())
df.drop('post_id',axis='columns',inplace=True)
# print(df.head())

# We first handle  The datetime columns here

df['post_datetime'] = pd.to_datetime(df['post_datetime'])
df['post_year'] = df['post_datetime'].dt.year
df['post_month'] = df['post_datetime'].dt.month
df['post_day'] = df['post_datetime'].dt.day
df['post_hour'] = df['post_datetime'].dt.hour
df['post_dayofweek'] = df['post_datetime'].dt.dayofweek
df.drop('post_datetime', axis=1, inplace=True)

# Now handle hashtags - extract number of hashtags

df['hashtag_count'] = df['hashtags'].apply(lambda x: len(str(x).split(',')) if pd.notnull(x) else 0)
df.drop('hashtags', axis=1, inplace=True)

# Now encode categorical columns
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

# For columns with few unique values, use OneHotEncoding

categorical_cols = ['platform', 'content_type', 'topic', 'language', 'region']

# oneHotEncoding
encoder = OneHotEncoder(sparse_output=False, drop='first')
encoded_array = encoder.fit_transform(df[categorical_cols])

# get feature names
encoded_features = encoder.get_feature_names_out(categorical_cols)

encoded_df = pd.DataFrame(encoded_array, columns=encoded_features, index=df.index)

df = df.drop(categorical_cols, axis=1)
df = pd.concat([df, encoded_df], axis=1)

# print(df.head())
# print(df.isna().sum())
# print(df.info())
# print(df.describe())

from sklearn.preprocessing import StandardScaler

# Identify numeric columns (excluding the target 'is_viral')
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
numeric_cols.remove('is_viral')  

scaler = StandardScaler()
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
# print(df.head())
x = df.drop('is_viral', axis=1)
y = df['is_viral']
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=42)
from sklearn.linear_model import LogisticRegression
 
model = LogisticRegression(max_iter=100)
model.fit(x_train,y_train)
print(model.score(x_test,y_test))
print(model.score(x_train,y_train))

#now check other performance metrics


from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns

y_pred = model.predict(x_test)
y_pred_proba = model.predict_proba(x_test)[:, 1]

print("\n=== Detailed Performance Metrics ===")
print(classification_report(y_test, y_pred))

# I take a help of ai for make this Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Not Viral', 'Viral'],
            yticklabels=['Not Viral', 'Viral'])
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion Matrix')
plt.show()

# now make a  ROC Curve 
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, 
         label=f'ROC curve (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.grid(True, alpha=0.3)
plt.show()

print(f"\nROC-AUC Score: {roc_auc:.4f}")
print(f"Precision (Viral): {cm[1,1]/(cm[0,1]+cm[1,1]):.4f}")
print(f"Recall (Viral): {cm[1,1]/(cm[1,0]+cm[1,1]):.4f}")