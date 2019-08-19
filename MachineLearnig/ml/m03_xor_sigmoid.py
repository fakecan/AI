from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import tensorflow as tf

# seed 값 생성
seed = 0
np.random.seed(seed)
tf.set_random_seed(seed)

# # 데이터 로드
# dataset = numpy.loadtxt(".\data\pima-indians-diabetes.csv", delimiter=",") # .: 현재 루트
# X = dataset[:,0:8]
# Y = dataset[:,8]

#1. 데이터
X = np.array([[0,0], [1,0], [0,1], [1,1]]) # and 모델
Y = np.array([0,1,1,0])

# 모델의 설정
model = Sequential()
model.add(Dense(16, input_dim=2, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# 모델 컴파일
model.compile(loss='binary_crossentropy', # 이진분류모델
                optimizer='adam',
                metrics=['accuracy'])

# 모델 실행
model.fit(X, Y, epochs=512, batch_size=10)

# 결과 출력
# print("\n Accuracy: %.4f" % (model.evaluate(X, Y)[1]))

loss, acc = model.evaluate(X, Y)
print("acc : ", acc)
print("loss : ", loss)
y_predict = model.predict(X)
print("y_predict :\n", y_predict)