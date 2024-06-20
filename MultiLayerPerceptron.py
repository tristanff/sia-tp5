import numpy as np
import random

def mse(expected, predicted):
    return np.mean(np.power(expected - predicted, 2))

def mse_derivative(expected, predicted):
    return 2 * (predicted - expected) / np.size(expected)

def predict(network, input):
    output = input
    for layer in network:
        output = layer.forward(output)
    return output

def predict_with_layer_value(network, input, layer_index):
    output = input
    values_at_layer = [output]
    for layer in network:
        output = layer.forward(output)
        values_at_layer.append(output)
    return output, values_at_layer[layer_index]

def train(network, error_function, error_derivative, x_train, y_train, epochs, verbose = True):

    mse = []
    max_index = len(x_train) - 1

    for e in range(epochs):
        error = 0
        
        for x, y in zip(x_train, y_train):
            # forward
            output = predict(network, x)

            # error
            error += error_function(y, output)

            # backward
            grad = error_derivative(y, output)

            for layer in reversed(network):
                grad = layer.backward(grad)
        
        
        error /= len(x_train)

        mse.append(error)
        if verbose:
            print(f"{e + 1}/{epochs}, error={error}")

    return mse

def train_with_max_error(network, error_function, error_derivative, x_train, y_train, max_epochs, max_error, verbose = True):

    mse = []

    epochs = 0
    for e in range(max_epochs):
        error = 0
        for x, y in zip(x_train, y_train):
            # forward
            output = predict(network, x)

            # error
            error += error_function(y, output)

            # backward
            grad = error_derivative(y, output)

            for layer in reversed(network):
                grad = layer.backward(grad)
        
        error /= len(x_train)

        mse.append(error)
        if verbose:
            print(f"{epochs + 1} epochs, error={error}")

        if error < max_error:
            break

        epochs += 1

    return mse, epochs

class Layer:
    def __init__(self):
        self.input = None
        self.output = None

    def forward(self, input):
        pass

    def backward(self, output_derivative):
        pass
    
class Dense(Layer):
    def __init__(self, input_size, output_size, learning_rate = 0.001, optimizer_type = None):
        self.weights = np.random.randn(output_size, input_size)
        self.bias =  np.random.randn(output_size, 1)
        self.learning_rate = learning_rate
        self.time_step = 0
        if optimizer_type =="ADAM":
            self.optimizer = AdamOptimizer(self.learning_rate)
        else:
            self.optimizer = GradientDescentOptimizer(self.learning_rate)

    def forward(self, input):
        self.input = input
        return np.dot(self.weights, self.input) + self.bias

    def backward(self, output_derivative):
        self.time_step += 1

        weights_gradient = np.dot(output_derivative, self.input.T)
        input_gradient = np.dot(self.weights.T, output_derivative)

        self.weights += self.optimizer.update(weights_gradient,self.time_step)

        # self.weights -= learning_rate * weights_gradient
        self.bias -= self.learning_rate * output_derivative
        return input_gradient

    
class Activation(Layer):
    def __init__(self, activation, activation_derivative):
        self.activation = activation
        self.activation_derivative = activation_derivative

    def forward(self, input):
        self.input = input
        return self.activation(self.input)

    def backward(self, output_gradient):
        return np.multiply(output_gradient, self.activation_derivative(self.input))


class Optimizer:
    def __init__(self, learning_rate):
        self.learningRate = learning_rate

    def update(self, gradient, time_step):
        pass

class AdamOptimizer(Optimizer):
    def __init__(self, learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8, shape=None):
        super().__init__(learning_rate)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        if shape is not None:
            self.m = np.zeros(shape)
            self.v = np.zeros(shape)
        else:
            self.m = None
            self.v = None

    def update(self, gradient, time_step):
        if self.m is None:
            self.m = np.zeros_like(gradient)
            self.v = np.zeros_like(gradient)
        
        self.m = self.beta1 * self.m + (1 - self.beta1) * gradient
        self.v = self.beta2 * self.v + (1 - self.beta2) * np.square(gradient)
        
        mHat = self.m / (1 - self.beta1**time_step)
        vHat = self.v / (1 - self.beta2**time_step)
        
        return (-self.learningRate * mHat) / (np.sqrt(vHat) + self.epsilon)
    

class GradientDescentOptimizer(Optimizer):
    def __init__(self, learning_rate):
        super().__init__(learning_rate)

    def update(self, gradient, time_step):
        return -self.learningRate * gradient