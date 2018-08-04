import numpy as np
import matplotlib.pyplot as plt
def sigmoid(x):
	return 1 / (1 + np.exp(-x))

def sigmoid_deriv(x):
	return x * (1-x)

x = np.linspace(-1, 1, 100)[:, np.newaxis] # shape (100, 1)
noise = np.random.normal(0, 0.1, size=x.shape)
y = x * 2 + noise

w1 = np.random.random((1, 10))
w2 = np.random.random((10, 1))
max_iter_size = 1000
iter_count = 0
sample_num, dim = x.shape
while iter_count < max_iter_size:
	net1 = np.dot(x, w1) # shape (100, 10)
	out1 = sigmoid(net1) # shape (100, 10)
	net2 = np.dot(out1, w2) # shape (100, 1)
	for j in range(sample_num):
		error = y[j] - net2[j]
		for k in range(10):
			w2[k] += 0.005 * error * out1[j][k]
			w1.T[k] += 0.1 * error * w2[k] * sigmoid_deriv(out1[j][k]) * x[j]
	loss = np.sum(np.power((y-net2), 2)) / (2*sample_num)
	print ("iter_count: ", iter_count, " the loss: ", loss)
	iter_count += 1

plt.scatter(x, y)
plt.plot(x, net2, 'r-')
plt.show()
