import tensorflow as tf
import random
# import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
tf.set_random_seed(777)
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)  # one_hot 처리

# hyper parameters
learning_rate = 0.001
training_epochs = 15
batch_size = 100

# input place holders
X = tf.placeholder(tf.float32, [None, 28*28])
X_img = tf.reshape(X, [-1, 28, 28, 1])  # img 28x28x1 (black/white)
Y = tf.placeholder(tf.float32, [None, 10])

# L1 ImgIn shape=(?, 28, 28, 1)
W1 = tf.Variable(tf.random_normal([3, 3, 1, 32], stddev=0.01))
            # kernel_size: (3,3), channel: 1(흑백), output: 32
print("W1: ", W1)   # shape=(3, 3, 1, 32)
#   Conv    ->   (?, 28, 28, 32)
#   Pool    ->   (?, 14, 14, 32)
L1 = tf.nn.conv2d(X_img, W1, strides=[1, 1, 1, 1], padding='SAME')  # stride: 몇 칸씩 움직일 것인가
#                                       [    ] 가운데 값 두개만 주로 쓴다. 바깥 2개는 거의 고정
print("L1: ", L1)   # shape=(?, 28, 28, 32)
L1 = tf.nn.relu(L1)
L1 = tf.nn.max_pool(L1, ksize=[1, 2, 2, 1], # (2, 2)로 자른 것을 2칸씩 이동 -> 반으로 줄어든다
                      strides=[1, 2, 2, 1], padding='SAME')
print("L1: ", L1)   # shape=(?, 14, 14, 32)

# L2 ImgIn shape=(?, 14, 14, 32)
W2 = tf.Variable(tf.random_normal([3, 3, 32, 64], stddev=0.01)) # W1의 32
print("W2: ", W2)   # shape=(3, 3, 32, 64)
#   Conv    ->   (?, 14, 14, 64)
#   Pool    ->   (?, 7, 7, 64)
L2 = tf.nn.conv2d(L1, W2, strides=[1, 1, 1, 1], padding='SAME')
print("L2: ", L2)   # shape=(?, 14, 14, 64)
L2 = tf.nn.relu(L2)
L2 = tf.nn.max_pool(L2, ksize=[1, 2, 2, 1],
                      strides=[1, 2, 2, 1], padding='SAME')
print("L2: ", L2)   # shape=(?, 7, 7, 64)                     
L2_flat = tf.reshape(L2, [-1, 7 * 7 * 64])
print("L2: ", L2)   # shape=(?, 7, 7, 64)               