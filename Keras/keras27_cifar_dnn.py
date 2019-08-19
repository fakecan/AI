from keras.datasets import cifar10
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.optimizers import SGD, Adam, RMSprop
import matplotlib.pyplot as plt

#CIFAR_10은 3채널로 구성된 32X32 이미지 60000장을 갖는다.
IMG_CHANNELS = 1
IMG_ROWS = 4
IMG_COLS = 1

#상수 정의
BATCH_SIZE = 128
NB_EPOCH = 20
NB_CLASSES = 10
VERBOSE = 1
VALIDATION_SPLIT = 0.2 #X_train과 Y_train 20프로 떼옴
OPTIM = RMSprop()


#데이터셋 불러오기
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
print('X_train shape : ', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# from sklearn.model_selection import train_test_split #비율 맞춰 잘라주는 함수 #열 분할
# x_train, x_test, y_train, y_test = train_test_split(
#     x, y, random_state = 66, test_size = 0.4
# )



#범주형으로 변환
Y_train = np_utils.to_categorical(y_train, NB_CLASSES)
Y_test = np_utils.to_categorical(y_test, NB_CLASSES)


# ...Scaler 사용해서 정규화
from sklearn.preprocessing import StandardScaler, MinMaxScaler
#scaler = StandardScaler()
scaler = MinMaxScaler() 

X_train = X_train.reshape(len(X_train) * 32 * 32, 3)
X_test = X_test.reshape(len(X_test) * 32 * 32, 3)

scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

X_train = X_train.reshape(50000, 32 * 32 * 3)
X_test = X_test.reshape(10000, 32 * 32 * 3)


#print(X_train.shape)
#print(X_test.shape)
#print(Y_train.shape)
#print(Y_test.shape)

import keras
td_hist = keras.callbacks.TensorBoard(log_dir='./graph', histogram_freq=0, #import 해준 것임 고로 
                                      write_graph=True, write_images=True)

#신경망 정의
model = Sequential()
model.add(Dense(100, input_shape=(IMG_ROWS, IMG_COLS, IMG_CHANNELS),
                activation = 'relu'))

model.add(Flatten()) #플래튼 이후는 DNN 모델화됨.
model.add(Dense(100))
model.add(Dropout(0.5))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(NB_CLASSES))

model.summary()

#학습
model.compile(loss='categorical_crossentropy', optimizer=OPTIM,
              metrics=['accuracy'])

history = model.fit(X_train, Y_train, batch_size=BATCH_SIZE,
                    epochs=NB_EPOCH, validation_split=VALIDATION_SPLIT,
                    verbose=VERBOSE)

print('Testing...')
score = model.evaluate(X_test, Y_test,
                       batch_size=BATCH_SIZE, verbose=VERBOSE)
print("\nTest score : ", score[0]) #loss
print("Test accuracy : ", score[1]) #acc


##모델 저장
#model_json = model.to_json()
#open('cifar10_architecture.json', 'w).write(model_json)
#model.save_

#히스토리에 있는 모든 데이터 나열
print(history.history.keys())


#단순 정확도에 대한 히스토리 요약
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc = 'upper left') #왼쪽 위에 그래프 선이 뭔지 알려주고 위에 플롯 순서와 대응된다
plt.show()

#손실에 대한 히스토리 요약
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc = 'upper left')
plt.show()