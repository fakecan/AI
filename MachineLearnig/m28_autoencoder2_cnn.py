# ■■■■■■■■■■ 데이터 ■■■■■■■■■■
from keras.datasets import mnist
import numpy as np
from keras.layers import Input, Dense, Dropout, Flatten, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model, Sequential
from keras.callbacks import ModelCheckpoint, EarlyStopping

(x_train, _), (x_test, _) = mnist.load_data()

x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
print(x_train.shape) # (60000, 28, 28, 1)
print(x_test.shape) # (10000, 28, 28, 1)

# ■■■■■■■■■■ 모델 구성 ■■■■■■■■■■
# 인코딩될 표현(representation)의 크기
img_shape = (28, 28, 1)

# 입력 플레이스홀더
input_img = Input(shape=img_shape)
# "encoded"는 입력의 인코딩된 표현
encoded = Conv2D(1, kernel_size=(28,28), padding='same',
                 activation='relu')(input_img)
            
# encoded = Conv2D(2, (3,3), activation='relu')(encoded)
# encoded = MaxPooling2D(pool_size=2)(encoded)
# encoded = UpSampling2D(size=(2, 2), data_format=None, interpolation='nearest')(encoded)
decoded = Conv2D(1, (28, 28), padding='same', activation='sigmoid')(encoded)

# 입력을 입력의 재구성으로 매핑할 모델
autoencoder = Model(input_img, decoded)

# 이 모델은 입력을 입력의 인코딩된 입력의 표현으로 매핑
encoder = Model(input_img, encoded)

# 인코딩된 입력을 위한 플레이스 홀더
encoded_input = Input(shape=(28, 28, 1)) # 디코딩의 인풋레이어로 시작
# 오토인코더 모델의 마지막 레이어 얻기
decoder_layer = autoencoder.layers[-1]
# 디코더 모델 생성
decoder = Model(encoded_input, decoder_layer(encoded_input))

# autoencoder.summary()
# encoder.summary()
# decoder.summary()

autoencoder.compile(optimizer='adam',
                    loss='binary_crossentropy', metrics=['accuracy'])

history = autoencoder.fit(x_train, x_train,
                          epochs=50, batch_size=256,
                          shuffle=True, validation_data=(x_test,x_test))

# 숫자들을 인코딩&디코딩
# test set에서 숫자들을 가져왔다는 것을 유의
encoded_imgs = encoder.predict(x_test)
decoded_imgs = decoder.predict(encoded_imgs)

print('encoded_imgs', encoded_imgs)
print('decoded_imgs', decoded_imgs)
print(encoded_imgs.shape) # (10000, 32)
print(decoded_imgs.shape) # (10000, 784)

# ■■■■■■■■■■ 이미지 출력 ■■■■■■■■■■
# Matplotlib 사용
import matplotlib.pyplot as plt

n = 10 # 몇 개의 숫자를 나타낼 것인지
plt.figure(figsize=(20,4))
for i in range(n):
    # 원본 데이터
    ax = plt.subplot(2, n, i+1)
    plt.imshow(x_test[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # 재구성된 데이터
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()

# ■■■■■■■■■■ 그래프 출력 ■■■■■■■■■■
def plot_acc(history, title=None):
    # summarize history for accuracy
    if not isinstance(history, dict):
        history = history.history

    plt.plot(history['acc'])
    plt.plot(history['val_acc'])
    if title is not None:
        plt.title(title)
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Training data', 'Validation data'], loc=0)
    # plt.show()

def plot_loss(history, title=None):
    # summarize history for accuracy
    if not isinstance(history, dict):
        history = history.history

    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    if title is not None:
        plt.title(title)
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Training data', 'Validation data'], loc=0)
    # plt.show()

plot_acc(history, '(a) 학습 경과에 따른 정확도 변화 추이')
plt.show()
plot_loss(history, '(b) 학습 경과에 따른 손실값 변화 추이')
plt.show()

loss, acc = autoencoder.evaluate(x_test, x_test)
print(loss, acc)