import os, cv2
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from numpy import array
from keras.utils import np_utils 
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from keras.optimizers import Adam
import matplotlib.pyplot as plt

# Parameters defination
CHANNELS = 1
ROWS = 28
COLUMNS = 28

BATCH_SIZE = 128
EPOCHS = 30
CLASSES = 10
VERBOSE = 1
# VALIDATION_SPLIT = 0.2
OPTIMIZER = Adam()

# ■■■■■■■■ train datasets ■■■■■■■■
TRAIN_DIR = './KerasImage/MNIST_data/train/'
train_folder_list = array(os.listdir(TRAIN_DIR))
# print(train_folder_list)
# ['0_zero' '1_one' '2_two' '3_three' '4_four' '5_five' '6_six' '7_seven' '8_eight' '9_nine']

train_images, train_labels = [], []

le = LabelEncoder()
# 문자열로 구성된 train_folder_list을 숫자형 리스트로 변환한다 
integer_encoded = le.fit_transform(train_folder_list)
# print(integer_encoded)  # [0 1 2 3 4 5 6 7 8 9]

# shape (10, ) -> shape (10, 1)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
oe = OneHotEncoder(sparse=False)
onehot_encoded = oe.fit_transform(integer_encoded)
# print(integer_encoded)

for index in range(len(train_folder_list)):
    path = os.path.join(TRAIN_DIR, train_folder_list[index])
    path = path + '/'
    img_list = os.listdir(path)
    # print(img_list)

    for img in img_list:
        img_path = os.path.join(path, img)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        train_images.append([np.array(img)])
        train_labels.append([np.array(onehot_encoded[index])])

train_images = np.reshape(train_images, (-1, 28*28))
train_labels = np.reshape(train_labels, (-1, 10))
train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32') / 255
print(train_images.shape)    # (42000, 28, 28, 1)
print(train_labels.shape)    # (42000, 10)
# train_images = np.array(train_images).astype(np.float32)
# train_labels = np.array(train_labels).astype(np.float32)
# np.save('./KerasImage/MNIST_train_data.npy', train_images)
# np.save('./KerasImage/MNIST_train_labels.npy', train_labels)

# ■■■■■■■■ test datasets ■■■■■■■■
TEST_DIR = './KerasImage/MNIST_data/test/'
test_folder_list = array(os.listdir(TEST_DIR))
 
test_images = []
test_labels = []
 
integer_encoded = le.fit_transform(test_folder_list)
 
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
onehot_encoded = oe.fit_transform(integer_encoded)
 
for index in range(len(test_folder_list)):
    path = os.path.join(TEST_DIR, test_folder_list[index])
    path = path + '/'
    img_list = os.listdir(path)
    for img in img_list:
        img_path = os.path.join(path, img)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        test_images.append([np.array(img)])
        test_labels.append([np.array(onehot_encoded[index])])
 
test_images = np.reshape(test_images, (-1, 28*28))
test_labels = np.reshape(test_labels, (-1, 10))
test_images = test_images.reshape(test_images.shape[0], 28, 28, 1).astype('float32') / 255
print(test_images.shape)    # (42000, 28, 28, 1)
print(test_labels.shape)    # (42000, 10)
# test_images = np.array(test_images).astype(np.float32)
# test_labels = np.array(test_labels).astype(np.float32)
# np.save("test_images.npy",test_images)
# np.save("test_labels.npy",test_labels)


# ■■■■■■■■ Model structure ■■■■■■■■
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same',
                 input_shape=(ROWS, COLUMNS, CHANNELS)))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(CLASSES, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer=OPTIMIZER, metrics=['accuracy'])

history = model.fit(train_images, train_labels,
                    batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=1)

print('---------- Testing... ----------')
loss, acc = model.evaluate(test_images, test_labels, batch_size=BATCH_SIZE, verbose=VERBOSE)
print("Test loss:", loss)
print("Test accuracy:", acc)

encoded_imgs = model.predict(test_images)
print(encoded_imgs)

n = 10 # 몇 개의 숫자를 나타낼 것인지
plt.figure(figsize=(20,4))
for i in range(n):
    # 원본 데이터
    ax = plt.subplot(2, n, i+1)
    plt.imshow(test_images[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # 재구성된 데이터
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(encoded_imgs[i].reshape(28, 28, 1))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()
