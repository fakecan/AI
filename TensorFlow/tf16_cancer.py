import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# ■■■■■■■■■■■■■■■■■■■■ Dataset Loading ■■■■■■■■■■■■■■■■■■■■
from sklearn.datasets import load_breast_cancer
dataset = load_breast_cancer()
x_data, y_data = dataset.data, dataset.target
# print(x_data.shape, y_data.shape) #(569, 30) (569,)

y_data = y_data.reshape(-1, 1)
# print(x_data.shape, y_data.shape) #(569, 30) (569, 1)

x_train, x_test, y_train, y_test = train_test_split(
    x_data, y_data, test_size = 0.2, random_state = 66
)

print(x_train.shape, x_test.shape)  # (455, 30) (114, 30)
print(y_train.shape, y_test.shape)  # (455, 1) (114, 1)

# ■■■■■■■■■■■■■■■■■■■■ Data Preprocessing ■■■■■■■■■■■■■■■■■■■■
def data_Preprocessing(train, test):
    for scaler in [MinMaxScaler()]:
        scaler.fit(train)
        train = scaler.transform(train)
        test = scaler.transform(test)
    return train, test

x_train, x_test = data_Preprocessing(x_train, x_test)

tf.set_random_seed(777)

# ■■■■■■■■■■■■■■■■■■■■ Model Design ■■■■■■■■■■■■■■■■■■■■
class Dense:

    node = 1
    Y, W, b = None, None, None

    def __init__(self, X, node, initialize=None, weight_name='w', input_dim=0):
        self.node = node
        # X가 Dense일 경우,
        if type(X) == type(self):
            input_dim = X.node
            tensor = X.Y
        # X가 Tensor일 경우,
        else:
            tensor = X

        # get_variable 사용 여부
        if initialize == None:
            self.W = tf.Variable(tf.random_normal([input_dim, node]))
            self.b = tf.Variable(tf.random_normal([node]))
        else:
            self.W = tf.get_variable(weight_name, shape = [input_dim, node], initializer = initialize)
            self.b = tf.Variable(tf.random_normal([node]))
        
        # Logic
        self.Y = tf.matmul(tensor, self.W) + self.b

    def relu(self):
        Y = self.Y
        self.Y = tf.nn.relu(Y)
        return self

    def softmax(self):
        Y = self.Y
        self.Y = tf.nn.softmax(Y)
        return self

    def sigmoid(self):
        Y = self.Y
        self.Y = tf.nn.sigmoid(Y)
        return self
    
    def dropout(self, keep_prob = 0.2):
        Y = self.Y
        self.Y = tf.nn.dropout(Y, keep_prob)
        return self

# input, output dimention
input_dim, output_dim = x_train.shape[1], 1
print(input_dim, output_dim)

# input Layer
X = tf.placeholder(tf.float32, [None, input_dim])
Y = tf.placeholder(tf.float32, [None, output_dim])

# Hidden Layer
hLayer = Dense(X, 1, input_dim = input_dim)
hLayer = hLayer.sigmoid()
hypothesis = hLayer.Y

cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1 - Y) * tf.log(1 - hypothesis))
train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

predicted = tf.cast(hypothesis > 0.5, dtype = tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype = tf.float32))

# fit, accuracy
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(10001):
        cost_val, _ = sess.run([cost, train], feed_dict={X: x_train, Y: y_train})
        if step % 200 == 0:
            print(step, cost_val)
    
    # Predict, Accuracy
    h, c, a = sess.run( [hypothesis, predicted, accuracy],
                        feed_dict={X: x_test, Y: y_test})
    print('Accuracy: ', a)