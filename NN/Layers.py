import numpy as np


class Layer:
    def __init__(self):
        self.input = None
        self.output = None

    def forward(self, inputs):
        pass

    def backward(self, gradient, hparams):
        pass


class DenseLayer(Layer):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.weights = np.random.randn(*(output_size, input_size)) * np.sqrt(2 / input_size)
        self.bias = np.zeros((output_size, 1))
        self.VdW = 0
        self.Vdb = 0

    def forward(self, inputs):
        self.input = inputs
        self.output = np.dot(self.weights, inputs) + self.bias
        return self.output

    def backward(self, gradient, hparams):
        dW = np.dot(gradient, self.input.T) / self.input.shape[1]
        db = (np.sum(gradient, axis=1) / self.input.shape[1]).reshape(-1, 1)

        if 'momentum' in hparams:
            momentum = hparams['momentum']
            self.VdW = momentum * self.VdW + (1-momentum)*dW
            self.Vdb = momentum * self.Vdb + (1-momentum)*db

            self.weights -= hparams['lr'] * self.VdW
            self.bias -= hparams['lr'] * self.Vdb
        else:
            self.weights -= hparams['lr'] * dW
            self.bias -= hparams['lr'] * db

        return np.dot(self.weights.T, gradient)
