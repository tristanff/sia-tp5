import numpy as np
import copy
from mlp import MLP, NoiseFunctionType
from loss import MSE
from activation_functions import Identity
from optimizers import Adam
from layer import Dense


class Autoencoder(MLP):
    def __init__(self, encoder=MLP(), decoder=MLP(), noise: NoiseFunctionType = None):
        super().__init__()
        self.layers += encoder.layers + decoder.layers
        self.encoder = encoder
        self.decoder = decoder
        self.noise = noise

    def predict(self, input_data):
        return self.encoder.predict(input_data)

    def train(self, dataset_input, dataset_test=None, loss=MSE(), epochs=1,
              callbacks=None, batchSize=1):
        if callbacks is None:
            callbacks = {}

        super().train(dataset_input, dataset_input, dataset_test, loss=loss, epochs=epochs, callbacks=callbacks,
                      autoencoder=True, noise=self.noise, batchSize=batchSize)

    def sampling(self, sampling_coordinates):
        return self.decoder.feedforward(sampling_coordinates)


class Sampler():
    def __init__(self, inputDim=1, outputDim=1, optimizer=Adam()):
        self.inputDim = inputDim
        self.outputDim = outputDim
        self.mean = Dense(self.inputDim, self.outputDim, activation=Identity(), optimizer=copy.copy(optimizer))
        self.logVar = Dense(self.inputDim, self.outputDim, activation=Identity(), optimizer=copy.copy(optimizer))

    def feedforward(self, input):
        self.latentMean = self.mean.feedforward(input)
        self.latentLogVar = self.logVar.feedforward(input)

        self.epsilon = np.random.standard_normal(size=(self.outputDim, input.shape[1]))
        self.sample = self.latentMean + np.exp(self.latentLogVar / 2.) * self.epsilon

        return self.sample

    def backpropagate(self, lastGradient):
        gradLogVar = {}
        gradMean = {}
        tmp = self.outputDim * lastGradient.shape[1]

        # KL divergence gradients
        gradLogVar["KL"] = (np.exp(self.latentLogVar) - 1) / (2 * tmp)
        gradMean["KL"] = self.latentMean / tmp

        # MSE gradients
        gradLogVar["MSE"] = 0.5 * lastGradient * self.epsilon * np.exp(self.latentLogVar / 2.)
        gradMean["MSE"] = lastGradient

        # backpropagate gradients thorugh self.mean and self.logVar
        return self.mean.backward(gradMean["KL"] + gradMean["MSE"]) + self.logVar.backward(
            gradLogVar["KL"] + gradLogVar["MSE"])

    def getKLDivergence(self, output):
        # output.shape[1] == batchSize
        return - np.sum(1 + self.latentLogVar - np.square(self.latentMean) - np.exp(self.latentLogVar)) / (
                    2 * self.outputDim * output.shape[1])


class VAE(MLP):

    def __init__(self, encoder=None, sampler=None, decoder=None):
        super().__init__()

        if encoder != None and sampler != None and decoder != None:
            self.layers = encoder.layers + [sampler.mean, sampler.logVar] + decoder.layers
            self.encoder = encoder
            self.sampler = sampler
            self.decoder = decoder
            self.decoder.loss = MSE()

    def feedforward(self, input, output_history=None):
        encoderOutput = self.encoder.feedforward(input, output_history)
        sample = self.sampler.feedforward(encoderOutput)
        if output_history is not None:
            output_history.append(sample)
        decoderOutput = self.decoder.feedforward(sample, output_history)

        return decoderOutput

    def backpropagate(self, output):
        self.decoder.backpropagate(output)
        decoderGradient = self.decoder.layers[0].gradient
        samplerGradient = self.sampler.backpropagate(decoderGradient)
        self.encoder.backpropagate(samplerGradient, useLoss=False)

    def train(self, dataset_input,  dataset_test=None, loss=MSE(), epochs=1, callbacks={}, batchSize=1):
        super().train(dataset_input=dataset_input, dataset_output=dataset_input, dataset_test=dataset_test,  loss=loss, epochs=epochs, callbacks=callbacks,
                      autoencoder=True, noise=None, batchSize=batchSize)

    def getLoss(self, output):
        return self.decoder.getLoss(output) + self.sampler.getKLDivergence(output)