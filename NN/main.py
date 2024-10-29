import numpy as np
import pandas as pd
import math

from Layers import DenseLayer
from Activations import *
from NNutils import *
from Losses import Crossentropy

# hyperparameters
hparams = {
    'lr': 0.01,
    'batch_size': 128,
    'epochs': 10,
    'lr_decay': 1,
    'momentum': 0.9,
}
BATCH_SIZE = hparams['batch_size']



df = pd.read_csv("mnist_train.csv")
data = df.to_numpy()
X = data[:,1:].T/255
y = data[:,0].reshape(1,data.shape[0])

X_train, X_test, y_train, y_test = train_test_split(X,y,train_size=0.8)

y_test = one_hot_encode(y_test)
y_train = one_hot_encode(y_train)



model = [
    DenseLayer(input_size=X_train.shape[0],output_size=128),
    Relu(),
    DenseLayer(input_size=128,output_size=64),
    Relu(),
    DenseLayer(input_size=64,output_size=16),
    Relu(),
    Dense_Softmax_CE(input_size=16, output_size=10),
]


# forward prop
for epoch in range(hparams['epochs']):
    epoch_cost = 0
    epoch_acc = 0
    for batch_num in range(int(math.ceil(X_train.shape[1]/BATCH_SIZE))):
        input = X_train[:,batch_num*BATCH_SIZE:(batch_num+1)*BATCH_SIZE]
        y_batch = y_train[:,batch_num*BATCH_SIZE:(batch_num+1)*BATCH_SIZE]
        for layer in model:
            input = layer.forward(input)
            # print(input)
        acc = get_accuracy(y_batch,input)
        cost = Crossentropy(y_batch,input)

        epoch_cost += cost
        epoch_acc += acc

        gradient = y_batch
        for layer in reversed(model):
            gradient = layer.backward(gradient,hparams)
            # print(gradient.shape)
    epoch_acc /= int(math.ceil(X_train.shape[1]/BATCH_SIZE))
    hparams['lr'] *= hparams['lr_decay']
    print("epoch:",epoch,"cost:",epoch_cost,"acc:",epoch_acc)