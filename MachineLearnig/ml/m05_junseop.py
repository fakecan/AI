import pandas as pd
from sklearn.model_selection import train_test_split 
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import accuracy_score
import numpy as np

# 붓꽃 데이터 읽어 들이기
iris_data = pd.read_csv("./data/iris.csv", encoding="UTF-8", names=['a','b','c','d','y'])
print(iris_data)
print(iris_data.shape)
print(type(iris_data))

# 붓꽃 데이터를 레이블과 입력 데이터로 분리하기
y = iris_data.loc[:, "y"]
#└ pandas의 series data는 내용을 보려면 values 이용
# ┗ y data들은 모두 문자열이므로 softmax로 분류가 불가능하므로 분류하기 위해서 문자열을 치환(oneHot Encoding)하여 표현해줘야한다
x = iris_data.loc[:, ["a",'b','c','d']]


# y2 = iris_data.iloc[:,4]
# x2 = iris_data.iloc[:36]

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
# lenc = LabelEncoder()
# y2 = lenc.fit_transform(y.values.reshape(-1,1)).toarray()
enc = OneHotEncoder()
y2 = enc.fit_transform(y.values.reshape(-1,1)).toarray()
# ┗ y의 data들은 정수값이 아니므로 keras.util.to_catagorical() 적용할 수 없다 -> OneHotEncoder 적용하게 되면 문자열을 bit표현으로 처리해준다

print(y[0], " -- one hot enocding --> ", y2[0])
print(y[50], " -- one hot enocding --> ", y2[50])
print(y[100], " -- one hot enocding --> ", y2[100])



# 학습 전용과 테스트 전용 분리하기
x_train, x_test, y_train, y_test = train_test_split(x,y2,test_size=0.2, train_size=0.8, shuffle=True)
# └ Iris Data 전체적으로 섞어주기 위해서 shuffle=True 주어 섞이게 한다

clf = Sequential()
clf.add(Dense(5, input_shape=(4, ), activation='relu'))
clf.add(Dense(8, activation='relu'))
clf.add(Dense(10, activation='relu'))
clf.add(Dense(3, activation='softmax'))

clf.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
clf.fit(x_train, y_train, epochs=300)

_, acc = clf.evaluate(x_test, y_test, batch_size=1)

# 평가하기
y_pred = clf.predict(x_test)
print("정답률 :", acc)       # 0.933 ~ 1.0
print("y_test :", y_test)
print("y_pred :", np.round(y_pred))