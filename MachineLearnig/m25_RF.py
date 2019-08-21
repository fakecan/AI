# RandomSearch, K-fold, CV
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, KFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from keras.utils import np_utils
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

# 데이터 읽어 들이기
wine = pd.read_csv("./data/winequality-white.csv", sep=";", encoding="utf-8")

# 데이터를 레이블과 데이터로 분리하기
y = wine["quality"]
x = wine.drop("quality", axis=1)

# y 레이블 변경하기
newlist = []
for v in list(y):
    if v <= 4:
        newlist += [0]
    elif v <= 7:
        newlist += [1]
    else:
        newlist += [2]
y = newlist

x_train, x_test, y_train, y_test = train_test_split(
    x, y, stratify=y, random_state=42
)

parameters = {'max_depth': [2, 4, 6, 8, 10, 12]
}

kfold_cv = KFold(n_splits=5, shuffle=True)

clf = RandomizedSearchCV(RandomForestClassifier(),
                         parameters,
                         cv=kfold_cv)

clf.fit(x_train, y_train)

print('최적의 매개 변수 =', clf.best_estimator_)

y_pred = clf.predict(x_test)
print('최종 정답률 = ', accuracy_score(y_test, y_pred))
last_score = clf.score(x_test, y_test)
print('최종 정답률 = ', last_score)