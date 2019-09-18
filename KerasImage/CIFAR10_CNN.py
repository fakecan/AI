from keras.datasets import cifar10
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.optimizers import Adam, RMSprop
import matplotlib.pyplot as plt

# Parameters defination
CHANNELS = 3
ROWS = 32
COLUMNS = 32

BATCH_SIZE = 128
EPOCHS = 60
CLASSES = 10
VERBOSE = 1
VALIDATION_SPLIT = 0.2
OPTIMIZER = RMSprop()

# Datasets load
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print('x_train shape:', x_train.shape, '|| y_train shape:', y_train.shape)
print('x_test shape:', x_test.shape, '|| y_test shape:', y_test.shape)
# x_train shape: (50000, 32, 32, 3) || y_train shape: (50000, 1)
# x_test shape: (10000, 32, 32, 3) || y_test shape: (10000, 1

# Categorical convert
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
# print(y_train.shape, y_test.shape)  # (50000, 10) (10000, 10)

# 실수형 지정 및 정규화
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255
# print(x_train.shape, x_test.shape)  (50000, 32, 32, 3) (10000, 32, 32, 3)

# ■■■■■■■■ Model structure ■■■■■■■■
model = Sequential()
# model = load_model("저장한 모델 파일")
'''
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', # padding='same',
                 input_shape=(ROWS, COLUMNS, CHANNELS)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(CLASSES, activation='softmax'))
'''
model.add(Conv2D(32, (3, 3), padding='same',
                 input_shape=(ROWS, COLUMNS, CHANNELS)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Flatten()) #플래튼 이후는 DNN 모델화됨.
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(CLASSES))
model.add(Activation('softmax'))

# model.summary()
# model.save('./KerasImage/save_cifar10_model.h5')

# ■■■■■■■■ Model learning ■■■■■■■■
model.compile(loss='categorical_crossentropy', optimizer=OPTIMIZER, metrics=['accuracy'])
history = model.fit(x_train, y_train,
                    batch_size=BATCH_SIZE, epochs=EPOCHS, validation_split=VALIDATION_SPLIT,
                    verbose=VERBOSE)

print('---------- Testing... ----------')
loss, acc = model.evaluate(x_test, y_test, batch_size=BATCH_SIZE, verbose=VERBOSE)
print("Test loss:", loss)
print("Test accuracy:", acc)

print(history.history.keys())

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'])
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'])
plt.show()