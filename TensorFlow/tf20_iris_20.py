import tensorflow as tf
import numpy as np
import pandas as pd
from keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

tf.set_random_seed(777)

# hyper parameters
learning_rate = 0.001
training_epochs = 100

# ■■■■■■■■■■■■■■■■■■■■ Dataset Loading ■■■■■■■■■■■■■■■■■■■■
iris_data = pd.read_csv('./data/iris.csv', encoding='utf-8',
                         names=['SepalLength','SepalWidth','PetalLength','PetalWidth','Name'])

y_data = iris_data.loc[:, 'Name']
x_data = iris_data.loc[:,['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth']]
print(x_data.shape, y_data.shape) # (150, 4) (150,)


# ■■■■■■■■■■■■■■■■■■■■ Data Splitting ■■■■■■■■■■■■■■■■■■■■
x_train, x_test, y_train, y_test = train_test_split(
    x_data, y_data, test_size=0.2, train_size=0.8, shuffle= True
)
print(x_train.shape, x_test.shape)  # (120, 4) (30, 4)
print(y_train.shape, y_test.shape)  # (120, 3) (30, 3)

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

# input place holders
X = tf.placeholder(tf.float32, [None, 4])
# X_img = tf.reshape(X, [-1, 4])
Y = tf.placeholder(tf.float32, [None, 3])

L1 = tf.layers.dense(X, 4, activation=tf.nn.relu)
# L1 = tf.layers.dropout(L1, 0.3)
L2 = tf.layers.dense(L1, 12, activation=tf.nn.relu)
L3 = tf.layers.dense(L2, 6, activation=tf.nn.relu)
logits = tf.layers.dense(L3, 3, activation=None)

# define cost/loss & optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=logits, labels=Y))
train = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# initialize
sess = tf.Session()
sess.run(tf.global_variables_initializer())

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(3001):
        cost_val, _ = sess.run([cost, train], feed_dict={X: x_train, Y: y_train})
        if step % 10 == 0:
            print(step, cost_val)

    print('Learning Finished!')

    # Test model and check accuracy
    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print('Accuracy:', sess.run(accuracy, feed_dict={X: x_test, Y: y_test}))
