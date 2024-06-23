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

""" Variational Autoencoder """
def reparameterize(mu, logvar):
    print('reparameterize input: ', mu, logvar, '\n')  # debugging
    if logvar is None: # debugging
        print('Error: No logvar. Check layer dimensions')

    std = np.exp(0.5 * logvar)
    eps = np.random.randn(*std.shape) # epsilon
    z = mu + eps * std
    print(z) # debugging
    return z

""" Variational Autoencoder """
def vae_loss(reconstructed, original, mu, logvar):
    print('vae_loss input: ', reconstructed, original, mu, logvar, '\n')  # debugging
    reconstruction_loss = np.mean(np.square(original - reconstructed))
    kl_divergence = -0.5 * np.sum(1 + logvar - np.square(mu) - np.exp(logvar))
    vae_loss = reconstruction_loss + kl_divergence
    print('vae_loss calc: ', vae_loss, '\n') # debug
    return vae_loss

""" Variational Autoencoder """
def train_vae(encoder, decoder, x_train, epochs, verbose=True):
    print('train_vae input: ','x_train shape: ',x_train.shape, '\n') # debugging
    losses = []

    for e in range(epochs):
        total_loss = 0

        for x in x_train:
            # Forward pass through encoder
            mu, logvar = None, None
            output = x
            for layer in encoder:
                print('train_vae output shape: ', output.shape, '\n')
                output = layer.forward(output)
                if isinstance(layer, Dense) and layer.output_type == 'mu':
                    mu = output
                elif isinstance(layer, Dense) and layer.output_type == 'logvar':
                    logvar = output

            # Reparameterize
            z = reparameterize(mu, logvar)
            print('z= ', z) # debug

            # Forward pass through decoder
            reconstructed = z
            for layer in decoder:
                reconstructed = layer.forward(reconstructed)

            # Compute loss
            loss = vae_loss(reconstructed, x, mu, logvar)
            total_loss += loss

            # Backward pass
            grad = 2 * (reconstructed - x) / np.size(x)
            for layer in reversed(decoder):
                grad = layer.backward(grad)

            grad_mu = mu / np.size(mu)
            grad_logvar = -0.5 * (1 - np.exp(logvar)) / np.size(logvar)
            for layer in reversed(encoder):
                if isinstance(layer, Dense) and layer.output_type == 'mu':
                    grad = layer.backward(grad_mu)
                elif isinstance(layer, Dense) and layer.output_type == 'logvar':
                    grad = layer.backward(grad_logvar)
                else:
                    grad = layer.backward(grad)

        total_loss /= len(x_train)
        losses.append(total_loss)

        if verbose:
            print(f"Epoch {e + 1}/{epochs}, Loss: {total_loss}")

    return losses

class Layer:
    def __init__(self):
        self.input = None
        self.output = None

    def forward(self, input):
        print('Layer.forward input: ', input, '\n')  # debugging
        pass

    def backward(self, output_derivative):
        pass

""" Variational Autoencoder """
class Dense_vae(Layer):
    def __init__(self, input_size, output_size, learning_rate=0.001, optimizer_type=None, output_type='standard'):
        self.weights = np.random.randn(output_size, input_size)
        self.bias = np.random.randn(output_size, 1)
        self.learning_rate = learning_rate
        self.time_step = 0
        self.output_type = output_type  # new attribute to handle output type
        if optimizer_type == "ADAM":
            self.optimizer = AdamOptimizer(self.learning_rate)
        else:
            self.optimizer = GradientDescentOptimizer(self.learning_rate)

    def forward(self, input):
        print(f'Dense_vae.forward input shape: {input.shape}')   # debugging
        self.input = input
        self.output = np.dot(self.weights, self.input) + self.bias
        return self.output

    def backward(self, output_derivative):
        print(f'Dense_vae.backward output_derivative shape: {output_derivative.shape}')  # debugging
        weights_gradient = np.dot(output_derivative, self.input.T)
        input_gradient = np.dot(self.weights.T, output_derivative)
        self.weights -= self.learning_rate * weights_gradient
        self.bias -= self.learning_rate * output_derivative
        return input_gradient

def train(network, error_function, error_derivative, x_train, y_train, epochs, verbose=True):
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

        mHat = self.m / (1 - self.beta1 ** time_step)
        vHat = self.v / (1 - self.beta2 ** time_step)

        return (-self.learningRate * mHat) / (np.sqrt(vHat) + self.epsilon)


class GradientDescentOptimizer(Optimizer):
    def __init__(self, learning_rate):
        super().__init__(learning_rate)

    def update(self, gradient, time_step):
        return -self.learningRate * gradient