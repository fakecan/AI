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

BATCH_SIZE = 32
EPOCHS = 20
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

x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255
# print(x_train.shape, x_test.shape)  (50000, 32, 32, 3) (10000, 32, 32, 3)

# Model structure
