# NeuralNetworks
Neural networks with simple code for understanding
## samples
`x = np.linspace(-1, 1, 100)[:, np.newaxis] # shape (100, 1)`
## add noise
`noise = np.random.normal(0, 0.1, size=x.shape)`
## labels
`y = x * 2 + noise`
## initialize weights
```
w1 = np.random.random((1, 10))
w2 = np.random.random((10, 1))
```
## build neural networks
### hidden layer
`net1 = np.dot(x, w1) # shape (100, 10)`
#### activity (sigmoid)
`out1 = sigmoid(net1) # shape (100, 10)`
### output layer
`net2 = np.dot(out1, w2) # shape (100, 1)`
## calculate error
`error = y[j] - net2[j]`
## back propagation
### output_layer -> hidden_layer
`w2[k] += 0.005 * error * out1[j][k]`
### hidden_layer -> input_layer
`w1.T[k] += 0.1 * error * w2[k] * sigmoid_deriv(out1[j][k]) * x[j]`
