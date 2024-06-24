from abc import ABC, abstractmethod
import numpy as np


class Optimizer(ABC):
    def __init__(self, learningRate=0.001):
        self.learningFactor = None
        self.learningRate = learningRate

    @abstractmethod
    def optimize(self):
        pass

    def setLearningFactor(self, batchSize):
        self.learningFactor = self.learningRate / batchSize


class Adam(Optimizer):

    def __init__(self, learningRate=0.001, beta_1=0.9, beta_2=0.999):
        super().__init__(learningRate)
        self.learningRate = learningRate
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = 1e-8
        self.m = None
        self.v = None
        self.t = 0

    def optimize(self, variable, variableGradient):
        if self.m is None:
            self.m = np.zeros(variableGradient.shape)
            self.v = np.zeros(variableGradient.shape)

        self.t = self.t + 1

        self.m = self.beta_1 * self.m + (1 - self.beta_1) * variableGradient  # Update biased first moment estimate
        self.v = self.beta_2 * self.v + (1 - self.beta_2) * np.square(
            variableGradient)  # Update biased second raw moment estimate

        m_hat = 1.0 / (1.0 - self.beta_1 ** self.t) * self.m  # Compute bias-corrected first moment estimate
        v_hat = 1.0 / (1.0 - self.beta_2 ** self.t) * self.v  # Compute bias-correct second raw moment estimate

        variable -= self.learningFactor * np.divide(m_hat, np.sqrt(v_hat) + self.epsilon)


class SGD(Optimizer):

    def __init__(self, learningRate=0.01, momentum=0):
        self.learningRate = learningRate
        self.momentum = momentum
        self.variable_update = None

    def optimize(self, variable, variableGradient):
        # initialize the update
        if self.variable_update is None:
            self.variable_update = np.zeros(variable.shape)
        # use momentum to update
        self.variable_update = self.momentum * self.variable_update + (1.0 - self.momentum) * variableGradient
        # update variable
        variable -= self.learningFactor * self.variable_update
