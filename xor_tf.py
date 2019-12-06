import tensorflow as tf

x_ = tf.placeholder(tf.float32, shape = [4, 2], name = 'x-input')
y_ = tf.placeholder(tf.float32, shape = [4, 1], name = 'y-input')

Theta1 = tf.Variable(tf.random_uniform([2,2], -1, 1), name = 'Theta1')
Theta2 = tf.Variable(tf.random_uniform([2, 1], -1, 1), name = 'Theta2')
Bias1 = tf.Variable(tf.zeros([2]), name = 'Bias1')
Bias2 = tf.Variable(tf.zeros([1]), name = 'Bias2')

A2 = tf.sigmoid(tf.matmul(x_, Theta1) + Bias1)
Hypothesis = tf.sigmoid(tf.matmul(A2, Theta2) + Bias2)
cost = tf.reduce_mean(((y_ * tf.log(Hypothesis)) + ((1 - y_) * tf.log(1.0 - Hypothesis))) * -1)
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cost)

x = [[0, 0], [0, 1], [1, 0], [1, 1]]
y = [[0], [1], [1], [0]]

init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

for i in range(100000):
	sess.run(train_step, feed_dict = {x_ : x, y_ : y})

	if (i % 10000 == 0):
		print('Epoch: ', i)
		print(sess.run(Hypothesis, feed_dict = {x_ : x, y_ : y}))
		print(sess.run(cost, feed_dict = {x_ : x, y_ : y}))