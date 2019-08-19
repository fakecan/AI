from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression #, Ridge, Lasso
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, LabelEncoder
from keras.utils import np_utils
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
import numpy as np
# 모델을 완성하시오.

boston = load_boston()
# boston = np.array(boston)
print(boston)
# print(boston.data.shape)
# print(boston.keys())
# print(boston.target)
print(boston.target.shape)

x = boston.data
y = boston.target()
x = np.array([x])
y = np.array([y])

scaler = StandardScaler()
scaler.fit(x)
x = scaler.transform(x)

print(type(boston))

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.4, train_size=0.6, shuffle=True
)

# 학습하기 
model = LinearRegression()
model.fit(x_train, y_train)

# 평가하기
y_pred = model.predict(x_test)
# print(classification_report(y_test, y_pred))
print("정답률 = ", accuracy_score(y_test, y_pred))
