# 실습
# iris.npy를 가지고 텐서플로 코딩을 하시오
# test와 train 분리할 것
# softmax?

import tensorflow as tf
import numpy as np
import pandas as pd
from keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# ■■■■■■■■■■■■■■■■■■■■ Dataset Loading ■■■■■■■■■■■■■■■■■■■■
iris_data = pd.read_csv('./data/iris.csv', encoding='utf-8',
                         names=['SepalLength','SepalWidth','PetalLength','PetalWidth','Name'])

y_data = iris_data.loc[:, 'Name']
x_data = iris_data.loc[:,['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth']]
# print(x.shape, y.shape) # (150, 4) (150,)

# ■■■■■■■■■■■■■■■■■■■■ Data Splitting ■■■■■■■■■■■■■■■■■■■■
x_train, x_test, y_train, y_test = train_test_split(
    x_data, y_data, test_size=0.2, train_size=0.8, shuffle= True
)

# ■■■■■■■■■■■■■■■■■■■■ Data Preprocessing ■■■■■■■■■■■■■■■■■■■■
def label_Encoding(train, test):    # LabelEncoder
    encoder = LabelEncoder()
    encoder.fit(train)
    train = encoder.transform(train)
    test = encoder.transform(test)
    
    return train, test

y_train, y_test = label_Encoding(y_train, y_test)
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

tf.set_random_seed(777)

# ■■■■■■■■■■■■■■■■■■■■ Model Design ■■■■■■■■■■■■■■■■■■■■
def input_Layer(x, input, output, keep_prob=0.2):
    W = tf.Variable(tf.random_normal([input, output]))
    b = tf.Variable(tf.random_normal([output]))

    equation = tf.matmul(x, W) + b
    y = tf.nn.relu(equation)

    if keep_prob > 0:
        y = tf.nn.dropout(y, keep_prob)
    return y

def hidden_Layer(x, input, output, keep_prob=0.2,
                 weight_name='W', optimizer='relu',
                 initialize=tf.contrib.layers.xavier_initializer(),
                 output_b = False   ):
    W = tf.get_variable(weight_name, shape=[input, output], initializer=initialize)
    b = tf.Variable(tf.random_normal([output]))

    equation = tf.matmul(x, W) + b
    if optimizer == 'relu':
        y = tf.nn.relu(equation)
    elif optimizer == 'softmax':
        y = tf.nn.softmax(equation)
        # y = tf.nn.softmax(y)    # ?
    
    if keep_prob > 0:   y = tf.nn.dropout(y, keep_prob)
    
    if output_b:    return y, W
    else:           return y
    
input_node = x_train.shape[-1]  # column value
output_node = y_train.shape[-1]
print(x_train.shape, y_train.shape)

# Input Layer
X = tf.placeholder(tf.float32, [None, input_node])
Y = tf.placeholder(tf.float32, [None, output_node])

# Hidden Layers
h_in = input_Layer(X, input_node, 15, 0)
h1 = hidden_Layer(h_in, 15, 25, 0.55, 'w1')
h2 = hidden_Layer(h1, 25, 30, 0.55, 'w2')
hypothesis, w = hidden_Layer(h2, 30, output_node, 0,
                            'w_out', 'softmax', output_b=True)

is_correct = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(hypothesis), axis = 1)) # 카테고리컬 크로스 엔트로피
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

# Test model
is_correct = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y, 1))
# Calculate accuracy
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(6001):
        _, cost_val = sess.run([optimizer, cost], feed_dict = {X: x_train, Y: y_train})
        if step % 100 == 0:
            print(step, cost_val)

    h, c, a = sess.run([hypothesis, is_correct, accuracy],
                        feed_dict={X: x_test, Y: y_test})
    print('Hypothesis:\n', h, '\nCorrect(y)\n', c, '\n Accuracy: ', a)