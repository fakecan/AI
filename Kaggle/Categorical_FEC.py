import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, RandomizedSearchCV, KFold
from sklearn.metrics import roc_auc_score

df_train = pd.read_csv('./data/cat-in-the-dat/train.csv')
df_test = pd.read_csv('./data/cat-in-the-dat/test.csv')
submission = pd.read_csv('./data/cat-in-the-dat/sample_submission.csv')
# y_predict = np.zeros(submission.shape[0])

# print('train data set: {} row and {} columns'.format(df_train.shape[0], df_train.shape[1]))
# print('test data set: {} row and {} columns'.format(df_test.shape[0], df_test.shape[1]))
# train data set: 300000 row and 25 columns
# test data set: 200000 row and 24 columns

# print(df_train.info())

x = df_train.drop(['target'], axis=1)
y = df_train['target']
# print(X.shape, y.shape)   # (300000, 24) (300000,)

# ■■■■■■■■■■■■■■■■ y distribution map ■■■■■■■■■■■■■■■■
# plt.bar(y.value_counts().index, y.value_counts())
# plt.gca().set_xticks([0, 1])
# plt.title('distribution of target variable')
# plt.show()
# ■■■■■■■■■■■■■■■■                    ■■■■■■■■■■■■■■■■
# ■■■■■■■■■■■■■■■■ def LogisticRegression ■■■■■■■■■■■■■■■■
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# def logistic(X, y):
#     x_train, x_test, y_train, y_test = train_test_split(X, y, random_state=66, test_size=0.2)
#     lr = LogisticRegression()
#     lr.fit(x_train, y_train)
#     y_predict = lr.predict(x_test)
#     print('Accuracy:', accuracy_score(y_test, y_predict))


# ■■■■■■■■■■■■■■■■              ■■■■■■■■■■■■■■■■

bin_dict = {'T':1, 'F':0, 'Y':1, N:0}


# ■■■■■■■■■■■■■■■■ LabelEncoder ■■■■■■■■■■■■■■■■
from sklearn.preprocessing import LabelEncoder
train = pd.DataFrame()
label = LabelEncoder()
for c in x.columns:
    if(x[c].dtype=='object'):
        train[c] = label.fit_transform(x[c])
    else:
        train[c] = x[c]

# print(train.head(3))
# ■■■■■■■■■■■■■■■■              ■■■■■■■■■■■■■■■■
# print('train data set: {} row and {} columns'.format(train.shape[0], train.shape[1])) # 300000 row and 24 columns
# logistic(train, y)  # 0.6904
# ■■■■■■■■■■■■■■■■ OneHotEncoder ■■■■■■■■■■■■■■■■
from sklearn.preprocessing import OneHotEncoder
onehot = OneHotEncoder()
onehot.fit(x)
train = onehot.transform(x)

# ■■■■■■■■■■■■■■■■               ■■■■■■■■■■■■■■■■
# print('train data set: {} row and {} columns'.format(train.shape[0], train.shape[1]))   # 300000 row and 316461 columns
# logistic(train, y)
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=66, test_size=0.2)

x_train = np.array(x_train)
y_train = np.array(y_train)
x_test = np.array(x_test)
y_test = np.array(x_test)

from xgboost import XGBClassifier
parameters = {'max_depth': [2, 4, 6, 8, 10, 12]}

kfold_cv = KFold(n_splits=5, shuffle=True)
clf = RandomizedSearchCV(XGBClassifier(),
                         param_distributions=parameters,
                         cv=kfold_cv)

# print('최적의 매개 변수 =', clf.best_estimator_)

clf.fit(x_train, y_train)

y_predict = clf.predict(x_test)
print('Accuracy:', roc_auc_score(y_test, y_predict))

# submission['target'] = y_predict.index

# submission = pd.DataFrame({
#     # 'id': test.id.values,
#     'target': y_predict
# })

# submission.to_csv('submission.csv', index=False)
