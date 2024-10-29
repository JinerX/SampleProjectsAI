import numpy as np
from Layers import *


class Activation(Layer):
    def __init__(self, activation, activation_derivative):
        super().__init__()
        self.activation = activation
        self.activation_derivative = activation_derivative

    def forward(self, input):
        self.input = input
        self.output = self.activation(self.input)
        return self.output

    def backward(self, gradient, hparams):
        return np.multiply(gradient, self.activation_derivative(self.input))


class Relu(Activation):
    def __init__(self):
        def ReLU(x):
            return x * (x > 0)

        def ReLU_derivative(x):
            return x > 0

        super().__init__(ReLU, ReLU_derivative)


class Dense_Softmax_CE(DenseLayer):
    def __init__(self, input_size, output_size):
        super().__init__(input_size, output_size)

        def Softmax(x):
            e_x = np.exp(x - np.max(x, axis=0, keepdims=True))
            return e_x / e_x.sum(axis=0, keepdims=True)

    def forward(self, input):
        self.input = input
        x = super().forward(self.input)
        self.output = self.Softmax(x)
        return self.output

    def backward(self, y_batch, hparams):
        return np.dot(self.weights.T,self.output - y_batch)

    def Softmax(self,x):
        e_x = np.exp(x - np.max(x, axis=0, keepdims=True))
        return e_x / e_x.sum(axis=0, keepdims=True)