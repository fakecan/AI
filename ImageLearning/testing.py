from keras.models import Sequential, load_model
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, BatchNormalization, Dropout
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import CSVLogger
import matplotlib.pyplot as plt
import numpy as np


# ■■■■■■■■ dataset load ■■■■■■■■
train_datagen = ImageDataGenerator(rescale=1./255,
                                   rotation_range=10,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.7,
                                   zoom_range=[0.9, 2.2],
                                   horizontal_flip=True,
                                   vertical_flip=True,
                                   fill_mode='nearest',
                                   validation_split=0.33)

training_set = train_datagen.flow_from_directory('./data/',
                                                 shuffle=True,
                                                 seed=13,
                                                 target_size=(224, 224),
                                                 batch_size=1,
                                                 class_mode='categorical',
                                                 subset='training')

validation_set = train_datagen.flow_from_directory('./data/',
                                                 shuffle=True,
                                                 seed=13,
                                                 target_size=(224, 224),
                                                 batch_size=10,
                                                 class_mode='categorical',
                                                 subset='validation')

# print(training_set.shape)
# print(type(training_set))
# print(validation_set.shape)
# print(type(validation_set))

training_set = np.array(training_set)
validation_set = np.array(validation_set)


csv_logger = CSVLogger('./log.csv', append=True, separator=';')


# ■■■■■■■■ model structure ■■■■■■■■
model = Sequential()
# model.add(Conv2D(32, (3, 3), padding='same', activation='relu', 
#                  input_shape=(224, 224, 3)))
# model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Flatten())
# model.add(Dense(128, activation='relu'))

model.add(Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(224, 224, 3)))
model.add(BatchNormalization(axis=3, scale=False))
model.add(MaxPooling2D(pool_size=(4, 4), strides=(4, 4), padding='same'))
model.add(Dropout(0.2))
 
model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
model.add(BatchNormalization(axis=3, scale=False))
model.add(MaxPooling2D(pool_size=(4, 4), strides=(4, 4), padding='same'))
model.add(Dropout(0.2))
 
model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
model.add(BatchNormalization(axis=3, scale=False))
model.add(MaxPooling2D(pool_size=(4, 4), strides=(4, 4), padding='same'))
model.add(Dropout(0.2))
 
model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
model.add(BatchNormalization(axis=3, scale=False))

model.add(Flatten())
model.add(Dropout(0.2)) 
model.add(Dense(512, activation='relu'))
model.add(Dense(5, activation='softmax'))



# ■■■■■■■■ model compile ■■■■■■■■
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])


# ■■■■■■■■ model fit ■■■■■■■■
hist = model.fit_generator(training_set,
                           steps_per_epoch=10,
                           epochs=1,
                           validation_data=validation_set,
                           validation_steps=10)

model.save('keras_model.h5')

# ■■■■■■■■ model evaluate ■■■■■■■■
print('>>>>>>>>>> Evaluate <<<<<<<<<<')
scores = model.evaluate_generator(validation_set, steps=10)
print("%s: %.2f%%" %(model.metrics_names[1], scores[1]*100))

# ■■■■■■■■ model predict ■■■■■■■■
print('>>>>>>>>>> Predict <<<<<<<<<<')
output = model.predict_generator(validation_set, steps=10)
np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
print(output)
# print(validation_set.class_indices) {'cheomseongdae': 0, 'colosseum': 1, 'damyang_metasequoia': 2, 'n_seoul_tower': 3, 'pyramid': 4}
# print(validation_set.filenames)


# 시각화
fig, loss_ax = plt.subplots()

acc_ax = loss_ax.twinx()

loss_ax.plot(hist.history['loss'], 'y', label='train loss')
loss_ax.plot(hist.history['val_loss'], 'r', label='val loss')
#loss_ax.set_ylim([0.0, 0.5])

acc_ax.plot(hist.history['acc'], 'b', label='train acc')
acc_ax.plot(hist.history['val_acc'], 'g', label='val acc')
#acc_ax.set_ylim([0.8, 1.0])

loss_ax.set_xlabel('epoch')
loss_ax.set_ylabel('loss')
acc_ax.set_ylabel('accuray')

loss_ax.legend(loc='upper left')
acc_ax.legend(loc='lower left')

plt.show()



n = 5 # 몇 개의 숫자를 나타낼 것인지
# plt.figure(figsize=(20,4))
for i in range(n):
    # 원본 데이터
    ax = plt.subplot(2, n, i+1)
    print(training_set[i].shape)
    plt.imshow(np.array(training_set[i]).reshape(224, 224, 3))
    # plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # 재구성된 데이터
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(output[i].reshape(224, 224, 3))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()

# plt.imshow(va)
# plt.imshow(decoded_imgs)

