import numpy as np

def sigmoid(x):
	return (1/(1 + np.exp(-x)))

def sigmoid_derivative(x):
	return (x*(1 - x))

x = np.array([[0, 0, 0, 1],
			 [0, 0, 1, 1],
			 [0, 1, 0, 1],
			 [0, 1, 1, 1],
			 [1, 0, 0, 1],
			 [1, 0, 1, 1]])
y = np.array([[0],
			 [1],
			 [1],
			 [0],
			 [1],
			 [0]])

weight_01 = np.random.random((4, 5))
weight_12 = np.random.random((5, 6))
weight_23 = np.random.random((6, 1))

for i in range(80000):
	l0 = x
	l1 = sigmoid(np.dot(x, weight_01))
	l2 = sigmoid(np.dot(l1, weight_12))
	l3 = sigmoid(np.dot(l2, weight_23))

	l3_error = y - l3
	l3_delta = l3_error*sigmoid_derivative(l3)
	l2_error = np.dot(l3_delta, weight_23.T)
	l2_delta = l2_error*sigmoid_derivative(l2)
	l1_error = np.dot(l2_delta, weight_12.T)
	l1_delta = l1_error*sigmoid_derivative(l1)

	weight_23 += np.dot(l2.T, l3_delta)
	weight_12 += np.dot(l1.T, l2_delta)
	weight_01 += np.dot(l0.T, l1_delta)

print(l3)