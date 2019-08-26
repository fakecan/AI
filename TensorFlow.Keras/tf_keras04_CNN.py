# from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Dropout
# from tensorflow.keras.models import Sequential

# model = Sequential()

# model.add(Conv2D(32, (5, 5), padding='valid',
#                         activation='relu', input_shape=(28, 28, 1)))
# model.add(MaxPool2D((2, 2)))
# model.add(Conv2D(64, (5, 5), padding='valid',
#                         activation='relu'))
# model.add(MaxPool2D((2, 2)))
# model.add(Flatten())
# model.add(Dense(1024, activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(10, activation='softmax'))


from tensorflow.keras import layers, models

model = models.Sequential()

model.add(layers.Conv2D(32, (5, 5), padding='valid',
                        activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPool2D((2, 2)))
model.add(layers.Conv2D(64, (5, 5), padding='valid',
                        activation='relu'))
model.add(layers.MaxPool2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(1024, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(10, activation='softmax'))

model.summary()

import time
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard

callback_list = [ModelCheckpoint(filepath='cnn_checkpoint.h5',
                                 monitor='val_loss',
                                 save_best_only=True),
                 TensorBoard(log_dir="logs/{}".format(time.asctime()))]