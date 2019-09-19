from icrawler.builtin import GoogleImageCrawler
import urllib3
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, load_model
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, BatchNormalization
from PIL import Image, ImageOps
import os, sys, glob
import numpy as np
import matplotlib.pyplot as plt

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning) # 경고 메세지 처리


attractionList = ['seoul tower night', 'cheomseongdae night', 'damyang metasequoia road', 'colosseum', 'pyramid']
attractionFolderList = ['n_seoul_tower', 'cheomseongdae', 'damyang_metasequoia', 'colosseum', 'pyramid']

for idx, val in enumerate(attractionList):
    google_crawler = GoogleImageCrawler(
        feeder_threads=10,
        parser_threads=10,
        downloader_threads=10,
        storage={'root_dir': 'image_data/' + attractionFolderList[idx]})
    google_crawler.session.verify = False
    filters = dict(type='photo')
    google_crawler.crawl(keyword=val, filters=filters, max_num=1000, file_idx_offset=0)


attractionList = ['seoul tower night', 'cheomseongdae night', 'damyang metasequoia road', 'colosseum', 'pyramid']
attractionFolderList = ['n_seoul_tower', 'cheomseongdae', 'damyang_metasequoia', 'colosseum', 'pyramid']
for attractionFolder in attractionFolderList:
    image_dir = 'image_data/' + attractionFolder + '/'
    target_resize_dir = './image-data/' + attractionFolder + '/'
    target_rotate_dir = './image-data_rotate/' + attractionFolder + '/'
    if not os.path.isdir(target_resize_dir):
        os.makedirs(target_resize_dir)
    if not os.path.isdir(target_rotate_dir):
        os.makedirs(target_rotate_dir)
    files = glob.glob(image_dir + "*.*")
    print(len(files))
    count = 1;
    size = (224, 224)
    for file in files:
        im = Image.open(file)
        im = im.convert('RGB')
        print("i: ", count, im.format, im.size, im.mode, file.split('/')[-1])
        count += 1
        im = ImageOps.fit(im, size, Image.ANTIALIAS, 0, (0.5, 0.5))
        im.save(target_resize_dir + file.split("/")[-1].split(".")[0] + ".jpg", quality=100)
        im.rotate(90).save(target_rotate_dir + "resize_" + file.split("/")[-1].split('.')[0] + ".jpg", quality=100)


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

training_set = train_datagen.flow_from_directory('/floyd/input/data',
                                                 shuffle=True,
                                                 seed=13,
                                                 target_size=(224, 224),
                                                 batch_size=15,
                                                 class_mode='categorical',
                                                 subset='training')

validation_set = train_datagen.flow_from_directory('/floyd/input/data',
                                                 shuffle=True,
                                                 seed=13,
                                                 target_size=(224, 224),
                                                 batch_size=10,
                                                 class_mode='categorical',
                                                 subset='validation')


# ■■■■■■■■ model structure ■■■■■■■■
model = Sequential()
model.add(Conv2D(32, (3, 3), padding='same', activation='relu', 
                 input_shape=(224, 224, 3)))
model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(5, activation='softmax'))


# ■■■■■■■■ model compile ■■■■■■■■
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])


# ■■■■■■■■ model fit ■■■■■■■■
hist = model.fit_generator(training_set,
                           steps_per_epoch=20,
                           epochs=1000,
                           validation_data=validation_set,
                           validation_steps=10)

# ■■■■■■■■ model evaluate ■■■■■■■■
print('>>>>>>>>>> Evaluate <<<<<<<<<<')
scores = model.evaluate_generator(validation_set, steps=10)


# ■■■■■■■■ model predict ■■■■■■■■
print('>>>>>>>>>> Predict <<<<<<<<<<')
output = model.predict_generator(validation_set, steps=10)
np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
print(output)
print(validation_set.filenames)


# ------------------------------------
# ------------------------------------
model = load_model('./?/.h5')

# ■■■■■■■■ model evaluate ■■■■■■■■
print('>>>>>>>>>> Evaluate <<<<<<<<<<')
scores = model.evaluate_generator(validation_set, steps=10)
print("%s: %.2f%" %(model.metrics_names[1], scores[1] * 100))

# ■■■■■■■■ model predict ■■■■■■■■
print('>>>>>>>>>> Predict <<<<<<<<<<')
output = model.predict_generator(validation_set, steps=10)
# print(output)
# print(validation_set.class_indices)
np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
# print(validation_set.filenames)

# 학습 과정 시각화
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