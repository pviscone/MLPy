"""
Implementation of the Multi Layer Perceptron.
"""

import time
import numpy as np
from numba import njit # <---- Decomment for a 5x speed up
from utils.losses import MEE_MSE
from neuron_CC import Neuron

class CCNN:
    """
    Cascade Correlation Neural Network class.
    """
    def __init__(self, w_init = 0.1, hidden_actv_f = 'sigmoid', output_actv_f='sigmoid',
                 eta=0.1, lamb=0, norm_L=2, alpha=0, nesterov=False):
        self.hid_neurons=[]
        self.out_neurons=[]
        # list of tuple with (function, parameter of funtcion)
        self.hidden_actv_f = (hidden_actv_f, 1)
        self.output_actv_f = (output_actv_f, 1)
        self.w_init=w_init # weights in range [-w_init, w_init] for each neu.
        self.eta = eta
        self.lamb = lamb
        self.norm_L = norm_L
        self.alpha = alpha
        self.nesterov = nesterov

        self.train_MEE = []
        self.train_MSE = []
        self.val_MEE = []
        self.val_MSE = []
        self.epoch = 0
        self.num_input = 0
        self.num_hidden = 0 
        self.transfer_line = None

    from utils_CC._add_hidden_neuron import add_hidden_neuron, jit_gradient
    from utils_CC._train import train, output_learning_step

    jit_gradient = staticmethod(jit_gradient)

    @property
    def out_net(self):
        return np.array([neu.out(self.transfer_line) for neu in self.out_neurons]).T

    def create_net(self, input_data, n_output):
        self.num_output = n_output
        self.transfer_line = input_data
        self.num_input = input_data.shape[1]
        for i in range(n_output):
            self.out_neurons.append(Neuron(data_shape = self.transfer_line.shape, 
                                      func = self.output_actv_f, 
                                      w_init = self.w_init))

    def feedforward(self):
        """ Move the input to the output of the net"""
        for i, neu_hidden in enumerate(self.hid_neurons, start = self.num_input):
            self.transfer_line[:, i] = neu_hidden.out(self.transfer_line[:,:i])

    def predict(self, data):
        n_data, n_features = data.shape
        self.transfer_line = np.ones((n_data, n_features + self.num_hidden))
        self.transfer_line[:, :n_features] = data
        self.feedforward()
        return self.out_net
