from keras.datasets import cifar10
from keras.utils import np_utils 
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Activation, Input
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.optimizers import SGD, Adam, RMSprop
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split,RandomizedSearchCV, GridSearchCV, KFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.pipeline import Pipeline
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Data load
(X_train, Y_train), (X_test, Y_test) = cifar10.load_data()
# print('X_train shape : ', X_train.shape) # (50000, 32, 32, 3)
# print('X_test shape : ', X_test.shape) # (10000, 32, 32, 3)
# print("Y_train shape : ", Y_train.shape) # (50000, 1)
# print('Y_test shape : ', Y_test.shape) # (10000, 1)

# Data split
X_train, _, Y_train, _ = train_test_split(X_train, Y_train, test_size=0.994)
print('Split.X_train shape : ', X_train.shape)
print('Split.Y_train shape : ', Y_train.shape)

# Reshape
X_train = X_train.reshape(X_train.shape[0], 32, 32, 3)
X_test = X_test.reshape(X_test.shape[0], 32, 32, 3)
# print('Reshape.X_train reshape : ', X_train.shape)
# print('Reshape.X_test reshape : ', X_test.shape)

# Categorical convert
Y_train = np_utils.to_categorical(Y_train) # (0~9) Onehot encoding
Y_test = np_utils.to_categorical(Y_test)
# print('Categorical.Y_train shape : ', Y_train.shape) # Ca.Y_train shape :  (300, 10) 
# print('Categorical.Y_test shape : ', Y_test.shape) # Ca.Y_test shape :  (10000, 10)

# Model
data_generator = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.02,
        height_shift_range=0.02,
        horizontal_flip=True
)

def build_network(keep_prob=0.1, optimizer='adam'):
    # CNN
    inputs = Input(shape=(32, 32, 3), name='input')
    x = Conv2D(32, kernel_size=(3,3), activation='relu', name='hidden1')(inputs)
    x = Conv2D(34, (3,3), activation='relu', name='hidden2')(x)
    x = MaxPooling2D(pool_size=2)(x)
    x = Conv2D(36, (3,3), activation='relu', name='hidden3')(x)
    x = Dropout(keep_prob)(x)
    x = Flatten()(x)
    x = Dense(120, activation='relu', name='hidden4')(x)
    x = Dropout(keep_prob)(x)    
    prediction = Dense(10, activation='softmax', name='output')(x)
   
    model = Model(inputs=inputs, outputs=prediction)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy',
                    metrics=['accuracy'])

    early_stopping_callback = EarlyStopping(monitor='acc', patience=128)
    model.fit_generator(data_generator.flow(X_train, Y_train, batch_size=32),
                              # steps_per_epoch=len(X_train) * 20 // 32, #몇배로 증폭시킬것이냐
                              steps_per_epoch = 20,
                              epochs=1,
                              # validation_data=(X_test, Y_test),
                              verbose=1, #callbacks=callbacks
                              callbacks=[early_stopping_callback]
                        )
    return model

def create_hyperparameters():
    batch_sizes = [10, 20, 30, 40, 50, 60]
    optimizers = ['rmsprop', 'adam', 'adadelta']
    dropouts = np.linspace(0.01, 0.1, 10)
    epochss = [50, 100, 150, 200, 250, 300]
    return {
            'model__batch_size':batch_sizes,
            'model__optimizer':optimizers,
            'model__keep_prob':dropouts,
            'model__epochs':epochss
            }

def di_scale(data):
    data = data.reshape(data.shape[0], 32*32*3)
    MinMaxScaler()
    data = data.reshape(data.shape[0], 32, 32, 3)

model = KerasClassifier(build_fn=build_network, verbose=1)
hyperparameters = create_hyperparameters()
pipe = Pipeline([("scaler", di_scale(X_train)), ('model', model)])
kfold_cv = KFold(n_splits=5, shuffle=True)

search = RandomizedSearchCV(estimator=pipe, param_distributions=hyperparameters,
                            n_iter=10, n_jobs=-1, verbose=1, cv=kfold_cv)

search.fit(X_train, Y_train)

print('Best parameter = ', search.best_params_)
print('Best estimator = ', search.best_estimator_)
print('Accuracy = ', search.score(X_test, Y_test))
