# ■■■■■■■■■■ 데이터 ■■■■■■■■■■
from keras.datasets import mnist
import numpy as np
from keras import regularizers
from keras.callbacks import EarlyStopping
from keras.layers import Input, Dense, Dropout, Flatten
from keras.models import Model

(x_train, _), (x_test, _) = mnist.load_data()

x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
print(x_train.shape) # (60000, 784)
print(x_test.shape) # (10000, 784)

# ■■■■■■■■■■ 모델 구성 ■■■■■■■■■■
# 인코딩될 표현(representation)의 크기
encoding_dim = 32
keep_prob = 0.11111111111111112

# 입력 플레이스홀더
input_img = Input(shape=(784, ))
# "encoded"는 입력의 인코딩된 표현
encoded = Dense(encoding_dim, activation='relu')(input_img)

encoded = Dense(144, activation='relu')(encoded) 
encoded = Dropout(keep_prob)(encoded)
encoded = Dense(128, activation='relu')(encoded) 

# encoded = Dense(64, activation='relu')(encoded) 
# encoded = Dropout(keep_prob)(encoded)
# encoded = Dense(64, activation='relu')(encoded)
# encoded = Dropout(keep_prob)(encoded)
# encoded = Dense(32, activation='relu')(encoded)
# encoded = Dense(16, activation='relu')(encoded)
# encoded = Dropout(keep_prob)(encoded)
# encoded = Dense(16, activation='relu')(encoded)

encoded = Dense(32, activation='relu')(encoded)
decoded = Dense(784, activation='sigmoid')(encoded)
# "decoded"는 입력의 손실 있는 재구성(lossy reconstruction)

# 입력을 입력의 재구성으로 매핑할 모델
autoencoder = Model(input_img, decoded) # 784 -> 32 -> 784

# 이 모델은 입력을 입력의 인코딩된 입력의 표현으로 매핑
encoder = Model(input_img, encoded)     # 784 -> 32

# 인코딩된 입력을 위한 플레이스 홀더
encoded_input = Input(shape=(encoding_dim, )) # 디코딩의 인풋레이어로 시작
# 오토인코더 모델의 마지막 레이어 얻기
decoder_layer = autoencoder.layers[-1]
# 디코더 모델 생성
decoder = Model(encoded_input, decoder_layer(encoded_input))    # 32 -> 784

# autoencoder.summary()
# encoder.summary()
# decoder.summary()

autoencoder.compile(optimizer='SGD',
                    loss='binary_crossentropy', metrics=['accuracy'])

early_stopping = EarlyStopping(monitor='loss', patience=60, mode='auto')
history = autoencoder.fit(x_train, x_train,
                          epochs=100, batch_size=10,
                          shuffle=True, validation_data=(x_test,x_test),
                          callbacks=[early_stopping])

loss, acc = autoencoder.evaluate(x_test, x_test)
print(loss, acc)