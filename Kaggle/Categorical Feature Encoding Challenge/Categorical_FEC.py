import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, RandomizedSearchCV, KFold

df_train = pd.read_csv('./data/cat-in-the-dat/train.csv')
df_test = pd.read_csv('./data/cat-in-the-dat/test.csv')
submission = pd.read_csv('./data/cat-in-the-dat/sample_submission.csv')

y_predict = np.zeros(submission.shape[0])
# print('train data set: {} row and {} columns'.format(df_train.shape[0], df_train.shape[1]))
# print('test data set: {} row and {} columns'.format(df_test.shape[0], df_test.shape[1]))
# train data set: 300000 row and 25 columns
# test data set: 200000 row and 24 columns

# print(df_train.info())

X = df_train.drop(['target'], axis=1)
y = df_train['target']
# print(X.shape, y.shape)   # (300000, 24) (300000,)

# ■■■■■■■■■■■■■■■■ y distribution map ■■■■■■■■■■■■■■■■
# plt.bar(y.value_counts().index, y.value_counts())
# plt.gca().set_xticks([0, 1])
# plt.title('distribution of target variable')
# plt.show()
# ■■■■■■■■■■■■■■■■                    ■■■■■■■■■■■■■■■■
# ■■■■■■■■■■■■■■■■ def logistic ■■■■■■■■■■■■■■■■
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
def logistic(X, y):
    x_train, x_test, y_train, y_test = train_test_split(
        X, y, random_state=66, test_size=0.2)
    lr = LogisticRegression()
    lr.fit(x_train, y_train)
    y_predict = lr.predict(x_test)
    print('Accuracy: ', accuracy_score(y_test, y_predict))
# ■■■■■■■■■■■■■■■■              ■■■■■■■■■■■■■■■■
# ■■■■■■■■■■■■■■■■ LabelEncoder ■■■■■■■■■■■■■■■■
from sklearn.preprocessing import LabelEncoder
train = pd.DataFrame()
label = LabelEncoder()
for c in X.columns:
    if(X[c].dtype=='object'):
        train[c] = label.fit_transform(X[c])
    else:
        train[c] = X[c]

# print(train.head(3))
# ■■■■■■■■■■■■■■■■              ■■■■■■■■■■■■■■■■
print('train data set: {} row and {} columns'.format(train.shape[0], train.shape[1])) # 300000 row and 24 columns
logistic(train, y)  # 0.6904
# ■■■■■■■■■■■■■■■■ OneHotEncoder ■■■■■■■■■■■■■■■■
from sklearn.preprocessing import OneHotEncoder
onehot = OneHotEncoder()
onehot.fit(X)
train = onehot.transform(X)

# ■■■■■■■■■■■■■■■■               ■■■■■■■■■■■■■■■■
print('train data set: {} row and {} columns'.format(train.shape[0], train.shape[1]))   # 300000 row and 316461 columns
logistic(train, y)  # Accuracy:  0.7622833333333333
