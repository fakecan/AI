from keras.models import Sequential
from keras.layers import Dropout, Flatten, BatchNormalization, Dense, LSTM, Conv2D 
import numpy as np
import tensorflow as tf
from keras import regularizers

# ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■ Dataset Loading ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
x_data = []
y_data = []

# ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■ Data Splitting ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x_data, y_data, random_state=0, train_size=0.8, test_size=0.2
)

# ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■ Data Preprocessing ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■

# ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■ Model Design ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
model = Sequential()
model.add(( , input_shape=(), activation=''))
model.add(Dropout())
model.add(Flatten())
model.add(BatchNormalization())
model.add(())

# model.summary()
# ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■ Model Compile ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
model.compile(optimizer='', loss='', metrics=[''])

# ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■ Model Fit ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
from keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='loss', patience=128, mode='auto')

model.fit(x_train, y_train, epochs=128, batch_size=128,
          callbacks=[early_stopping], validation_data=(x_val, y_val))

# ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■ Model Evaluate&Predict ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
loss, acc = model.evaluate(x_test, y_test, batch_size=128)
print("Accuracy: ", acc)

y_predict = model.predict(x_test)
print(y_predict)

# RMSE formula
from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
print("RMSE : ", RMSE(y_test, y_predict))

# R2 formula
from sklearn.metrics import r2_score
r2_y_predict = r2_score(y_test, y_predict)
print("R2 : ", r2_y_predict)

# #RMAE formula
# from sklearn.metrics import mean_absolute_error
# def RMAE(y_test, y_predict):
#     return np.sqrt(mean_absolute_error(y_test, y_predict))
# print("RMAE : ", RMAE(y_test, y_predict))
