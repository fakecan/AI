from keras.models import Sequential

# filter_size = 32
# kernel_size = (3, 3)

# model.add(Conv2D(filter_size, kernel_size, #filter_size: output, kernel_size: 몇 개 자를 것인가
#                  input_shape = (5, 5, 1))) #가로28 세로28 흑백1, 컬러였으면 3
# model.add(Conv2D(7, (2,2), padding = 'same', input_shape = (5, 5, 1)))
# #4x4x7 가세5,5흑백을 (2,2)로 4x4 잘라서 그것을 7장을 뽑아낸다. 4x4x7
# #35는 (2x2+1(bias)) x 7 params

#필터 사이즈는 커널 사이즈로 자른 데이터를 32장 만들어낸다.
#padding은 same과 valid가 있는데 same은 그대로 뽑아냄
#default 값은 valid
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense #MaxPooling2D 중복없이 자른다 #Flatten 데이터를 쫙 펴준다. Dropout도 가능
model = Sequential()


model.add(Conv2D(7, (2,2), padding = 'same', input_shape = (5, 5, 1))) #가로 세로 컬러
model.add(Conv2D(16, (2, 2), padding = 'same'))

model.add(MaxPooling2D(2,2))
# model.add(Flatten())
# model.add(Dense(2, activation='relu'))
model.add(Conv2D(8, (2, 2)))


model.summary()

# _________________________________________________________________
# Layer (type)                 Output Shape              Param #
# =================================================================
# conv2d_1 (Conv2D)            (None, 4, 4, 7)           35
# _________________________________________________________________
# conv2d_2 (Conv2D)            (None, 2, 2, 16)          1024
# _________________________________________________________________
# conv2d_3 (Conv2D)            (None, 1, 1, 8)           520
# =================================================================
# Total params: 1,579
# Trainable params: 1,579
# Non-trainable params: 0
# _________________________________________________________________

# _________________________________________________________________
# Layer (type)                 Output Shape              Param #
# =================================================================
# conv2d_1 (Conv2D)            (None, 4, 4, 7)           35
# _________________________________________________________________
# conv2d_2 (Conv2D)            (None, 3, 3, 16)          464
# _________________________________________________________________
# conv2d_3 (Conv2D)            (None, 2, 2, 8)           520
# =================================================================
# Total params: 1,019
# Trainable params: 1,019
# Non-trainable params: 0
# _________________________________________________________________