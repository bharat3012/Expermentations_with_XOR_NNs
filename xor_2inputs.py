import numpy as np

def sigmoid(x):
	return (1/(1+np.exp(-x)))
def sigmoid_deriv(x):
	return (x*(1 - x))

x = np.array([[0,0,1],
              [0,1,1],
              [1,0,1],
              [1,1,1]])
y = np.array([[0],
              [1],
              [1],
              [0]])

weight_01 = np.random.random((3, 4))
weight_12 = np.random.random((4, 1))

for i in range(60000):
	l0 = x
	l1 = sigmoid(np.dot(l0, weight_01))
	l2 = sigmoid(np.dot(l1, weight_12))

	l2_error = y - l2
	l2_delta = l2_error*sigmoid_deriv(l2)
	l1_error = l2_delta.dot(weight_12.T)
	l1_delta = l1_error*sigmoid_deriv(l1)

	weight_12 += l1.T.dot(l2_delta)
	weight_01 += l0.T.dot(l1_delta)

print(l2)