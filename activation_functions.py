import numpy as np
from MultiLayerPerceptron import Activation

class Tanh(Activation):
    def __init__(self):
        # Tanh activation function
        tanh = lambda x: np.tanh(x)
        # Derivative of tanh activation function
        tanh_derivative = lambda x: 1 - np.tanh(x) ** 2
        super().__init__(tanh, tanh_derivative)

class Mish(Activation):
    def __init__(self):
        # Mish activation function
        mish = lambda x: x * np.tanh(x)
        # Derivative of mish activation function
        mish_derivative = lambda x: np.tanh(x) + x * (1 / np.cosh(x))**2
        super().__init__(mish, mish_derivative)

class Sigmoid(Activation):
    def __init__(self):
        # Sigmoid activation function
        def sigmoid(x):
            return 1 / (1 + np.exp(-x))

        # Derivative of sigmoid activation function
        def sigmoid_derivative(x):
            s = sigmoid(x)
            return s * (1 - s)

        super().__init__(sigmoid, sigmoid_derivative)

class Linear(Activation):
    def __init__(self):
        # Linear activation function
        def linear(x):
            return x

        # Derivative of linear activation function
        def linear_derivative(x):
            return 1

        super().__init__(linear, linear_derivative)

class Identity(Activation):
    def __init__(self ):
        # Identity activation function
        def identity(x):
            return x

        # Derivative of identity activation function
        def identity_derivative(x):
            return np.ones(x.shape)
        
        super().__init__(identity,identity_derivative)

class ReLU(Activation):
    def __init__(self ):
        # ReLU activation function
        def apply(x):
            return x * (x > 0)

        # Derivative of ReLU activation function
        def derivative(x):
            return 1. * (x > 0)

        super().__init__(apply,derivative)
