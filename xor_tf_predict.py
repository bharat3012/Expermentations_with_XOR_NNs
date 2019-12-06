import numpy as np
import tensorflow as tf

x_ = tf.placeholder(tf.float32, shape = [None, 2])
y_ = tf.placeholder(tf.float32, shape = [None, 1])

x_.shape
W1 = tf.Variable(tf.random_uniform(shape = [2, 10]))
b1 = tf.Variable(tf.random_uniform(shape = [10]))
a1 = tf.sigmoid(tf.matmul(x_, W1) + b1)

W2 = tf.Variable(tf.random_uniform(shape = [10, 4]))
b2 = tf.Variable(tf.random_uniform(shape = [4]))
a2 = tf.nn.relu(tf.matmul(a1, W2) + b2)

W3 = tf.Variable(tf.random_uniform(shape = [4, 1]))
b3 = tf.Variable(tf.random_uniform(shape = [1]))
a3 = tf.sigmoid(tf.matmul(a2, W3) + b3)

cost = tf.reduce_mean(((y_ * tf.log(a3)) + ((1 - y_) * tf.log(1.0 - a3))) * -1)

train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cost)

x = [[0, 0], [0, 1], [1, 0], [1, 1]]
y = [[0], [1], [1], [0]]

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

for i in range(50000):
	sess.run(train_step, feed_dict = {x_ : x, y_ : y})

print(sess.run(a3, feed_dict = {x_ : x, y_ : y}))
print(sess.run(cost, feed_dict = {x_ : x, y_ : y}))

print(sess.run(a3, feed_dict = {x_ : [[0, 0], [0, 1]]}))