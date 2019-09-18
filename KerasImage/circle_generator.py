import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

np.random.seed(3)

train_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(
    './KerasImage/handwriting_shape/train/',
    target_size=(24, 24),
    batch_size=3,
    class_mode='categorical')
# Found 45 images belonging to 3 classes.

test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    './KerasImage/handwriting_shape/test',
    target_size=(24, 24),
    batch_size=3,
    class_mode='categorical')
# Found 15 images belonging to 3 classes.

# ■■■■■■■■ Model structure ■■■■■■■■
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', # padding='same',
                 input_shape=(24, 24, 3)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(3, activation='softmax'))

# model.summary()

# model.save('./KerasImage/.h5')

# ■■■■■■■■ Model learning ■■■■■■■■
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit_generator(
    train_generator,
    steps_per_epoch=15,
    epochs=50,
    validation_data=test_generator,
    validation_steps=5)

score = model.evaluate_generator(test_generator, steps=5)
print("%s: %.2f%%" %(model.metrics_names[1], score[1]*100))

# predict
print("---------- Predict ----------")
output = model.predict_generator(test_generator, steps=5)
np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
print(test_generator.class_indices)
print(output)
print(test_generator.filenames)

# from IPython.display import SVG
# from keras.utils.vis_utils import model_to_dot
# %matplotlib inline
# SVG(model_to_dot(model, show_shapes=True).create(prog='dot', format='svg'))
# plt.show()

