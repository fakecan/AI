import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM, Activation
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC, SVC

# 붓꽃 데이터 읽어들이기
iris_data = pd.read_csv('./data/iris.csv', encoding='utf-8',
                        names=['a', 'b', 'c', 'd', 'y']) #, header=None)
# print(iris_data)
print(iris_data.shape)
print(type(iris_data))

# iris_data = iris_data.astype('float32')
# try: float(np.array(iris_data))
# except ValueError: pass 
# iris_data[i] = [float(x) for x in iris_data[i] if x != ''] 

# 붓꽃 데이터를 레이블과 입력 데이터로 분리하기
y = iris_data.loc[:, "y"]
x = iris_data.loc[:,["a", "b", "c", "d"]]

# y2 = iris_data[:,4]
# x2 = iris_data.iloc[:, 0:4]

# print("====================")
# print(x.shape) # (150, 4)
# print(y.shape) # 

enc = OneHotEncoder()
y2 = enc.fit_transform(y.values.reshape(-1,1)).toarray()
# ┗ y의 data들은 정수값이 아니므로 keras.util.to_catagorical() 적용할 수 없다 -> OneHotEncoder 적용하게 되면 문자열을 bit표현으로 처리해준다

print(y[0], " -- one hot enocding --> ", y2[0])
print(y[50], " -- one hot enocding --> ", y2[50])
print(y[100], " -- one hot enocding --> ", y2[100])

# 학습 전용과 테스트 전용 분리하기
x_train, x_test, y_train, y_test = train_test_split(
    x, y2, test_size=0.2, train_size=0.8, shuffle=True
    )

# print(y_test) # str 형식으로 저장되어 있다

# print(x_train.shape)
# print(x_test.shape)

# 모델
clf = Sequential()

clf.add(Dense(8, input_shape=(4, ), activation='relu'))
clf.add(Dense(8))
clf.add(Dense(8))
clf.add(Dense(8))
clf.add(Dense(3))
clf.add(Activation('softmax'))

#3. 실행
clf.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

clf.fit(x_train, y_train, epochs=200, batch_size=2)

#4. 평가 및 예측
acc = clf.evaluate(x_test, y_test)
# print("acc = ", acc)


# 학습하기
# clf = KNeighborsClassifier(n_neighbors=1)
# clf = SVC()
# clf = LinearSVC()
# clf.fit(x_train, y_train)

# 평가하기
y_pred = clf.predict(x_test)
print("정답률 : ", acc) # 0.933~1.0
