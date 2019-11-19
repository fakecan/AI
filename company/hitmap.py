from keras.applications.vgg16 import VGG16
from keras import backend as K
import matplotlib.pyplot as plt 
import os

model = VGG16(weights='imagenet')

from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input, decode_predictions
import numpy as np

name_path = 'rabbit9'
img_path = './datasets/'+name_path+'.jpg' # 이미지 경로

img = image.load_img(img_path, target_size=(224, 224))

x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

preds = model.predict(x)
print("Predicted : ", decode_predictions(preds, top=3)[0])

print(np.argmax(preds[0]))

animal_output = model.output[:, np.argmax(preds[0])]

last_conv_layer = model.get_layer('block5_conv3') # VGG16 마지막 합성곱 층
grads = K.gradients(animal_output, last_conv_layer.output)[0]
pooled_grads = K.mean(grads, axis=(0, 1, 2))

iterate = K.function([model.input], [pooled_grads, last_conv_layer.output[0]])

pooled_grads_value, conv_layer_output_value = iterate([x])

for i in range(512):
    conv_layer_output_value[:, :, i] *= pooled_grads_value[i]

heatmap = np.mean(conv_layer_output_value, axis=-1)

heatmap = np.maximum(heatmap, 0)
heatmap /= np.max(heatmap)
# plt.matshow(heatmap)
plt.imshow(heatmap)
plt.show()

import cv2

img = cv2.imread(img_path)
heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
heatmap = np.uint8(255 * heatmap)
heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
plt.imshow(heatmap)
plt.show()


superimposed_img = heatmap * 0.5 + img

cv2.imwrite('./datasets_cam/'+name_path+'_cam.jpg', superimposed_img)
img_pre = cv2.imread('./datasets_cam/'+name_path+'_cam.jpg')
cv2.imshow('gray', img_pre)
cv2.waitKey(0)
cv2.destroyAllWindows()

