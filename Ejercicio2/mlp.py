import numpy as np
import logging

from loss import MSE
from typing import Callable

NoiseFunctionType = Callable[[np.ndarray], np.ndarray]

class MLP:
    def __init__(self):
        self.layers = []
        self.loss = None

    def addLayer(self, layer):
        self.layers.append(layer)

    def feedforward(self, input_data, output_history=None):
        for layer in self.layers:
            input_data = layer.feedforward(input_data)
            if output_history is not None:
                output_history.append(input_data)

        return input_data

    def predict(self, input_data):
        input_data = input_data.reshape((input_data.shape[0], 1))
        return self.feedforward(input_data)

    def backpropagate(self, output, useLoss=True, updateParameters=True):
        if useLoss:
            # step 1:
            lastGradient = self.loss.derivative(output, self.layers[-1].a) * self.layers[-1].activation.derivative(
                self.layers[-1].z)
            # step 2:
            isOutputLayer = True
            for layer in self.layers[::-1]:
                lastGradient = layer.backward(lastGradient, outputLayer=isOutputLayer,
                                              updateParameters=updateParameters)
                isOutputLayer = False

        else:
            isOutputLayer = False
            lastGradient = output
            for layer in self.layers[::-1]:
                lastGradient = layer.backward(lastGradient, outputLayer=isOutputLayer,
                                              updateParameters=updateParameters)

    def train(self, dataset_input, dataset_output, dataset_test=None, loss=MSE(), epochs=1, metrics=None, callbacks={},
              autoencoder=False, noise: NoiseFunctionType = None, batchSize=1):

        self.loss = loss

        # set batch size before training
        for layer in self.layers:
            layer.setBatchSize(batchSize)

        ind = 0  # number of samples processed
        for i in range(epochs):
            for j in range(len(dataset_input)):

                if noise is not None:
                    noisy_input = noise(dataset_input[j])
                    input_reshaped = np.reshape(noisy_input, (len(noisy_input), batchSize))
                else:
                    input_reshaped = np.reshape(dataset_input[j], (len(dataset_input[j]), batchSize))

                output_reshaped = np.reshape(dataset_output[j], (len(dataset_output[j]), batchSize))

                self.feedforward(input_reshaped)
                self.backpropagate(output_reshaped)

                if ind % 1000 < batchSize:
                    if dataset_test:
                        self.validate(dataset_test, ind, callbacks, batchSize=batchSize)

                ind += batchSize

    def validate(self, dataset_test, ind, callbacks, batchSize=1):
        rand_ind = np.random.randint(0, len(dataset_test))
        test_reshaped = np.reshape(dataset_test[rand_ind], (len(dataset_test[rand_ind]), batchSize))
        self.feedforward(test_reshaped)


    def getLoss(self, label):
        return self.loss.apply(label, self.layers[-1].a)


    def getAccuracy(self, label):
        difference = np.argmax(self.layers[-1].a, axis=0) - np.argmax(label, axis=0)
        accuracy = (1 - np.count_nonzero(difference) / len(difference)) * 100
        return accuracy



