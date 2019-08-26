import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, Flatten
from keras.callbacks import EarlyStopping

# ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■ Dataset Loading ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
from keras.datasets import mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
print(train_images.shape, test_images.shape)    # (60000, 28, 28) (10000, 28, 28)
print(train_labels.shape, test_labels.shape)    # (60000,) (10000,)

# ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■ to_categorical ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
from keras.utils import to_categorical
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

# ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■ Model Design ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
model = Sequential()
model.add(LSTM(128, input_shape=(28, 28), return_sequences=True))
# model.add(Dropout(0.2))
model.add(LSTM(64, activation='relu'))
model.add(Dense(32)) 
model.add(Dense(24))
model.add(Dense(16))
# model.add(Dropout(0.2))
model.add(Dense(10, activation='softmax'))

# ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■ Model Compile ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■ Model Fit ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
early_stopping = EarlyStopping(monitor='loss', patience=20, mode='auto')
model.fit(train_images, train_labels, epochs=20, batch_size=128,
          callbacks=[early_stopping])

# ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■ Model Evaluate&Predict ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
loss, acc = model.evaluate(test_images, test_labels, batch_size=128)
print("acc : ", acc)

y_predict = model.predict(test_images) #x는 훈련시킨 값
print(y_predict)

# #RMSE 구하기
# from sklearn.metrics import mean_squared_error #레거시한 머신 러닝 중 하나
# def RMSE(y_test, y_predict):
#     return np.sqrt(mean_squared_error(y_test, y_predict))
# print("RMSE : ", RMSE(test_labels, y_predict))

# # R2 구하기
# from sklearn.metrics import r2_score
# r2_y_predict = r2_score(test_labels, y_predict)
# print("R2 : ", r2_y_predict)
