"""
Implementation of the Cascade Correlation Neural Network.
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
    def __init__(self, w_init = 0.1, hidden_actv_f = 'sigmoid', output_actv_f='sigmoid'):
        """
        __init__ function of Cascade Correlation Neural Network.

        Parameters
        ----------
        hidden_actv_f : tuple or string
            If string just set the hidden actv funcs as (hidden_actv_f, 1),
            where the first argument is the activation function, the second is 
            the parameter of this function (exponential decay for the sigmoid, 
            angular coefficient for linear).
            Otherwise just set the the tuple as hidden_actv_f. Function avaible
            are:
                - sigmoid
                - linear
                - relu
        output_actv_f : tuple or string
            If string just set the output actv funcs as (output_actv_f, 1),
            where the first argument is the activation function, the second is 
            the parameter of this function (exponential decay for the sigmoid, 
            angular coefficient for linear).
            Otherwise just set the the tuple as output_actv_f. Function avaible
            are:
                - sigmoid
                - linear
                - relu
        """
        self.hid_neurons=[]
        self.out_neurons=[]

        # list of tuple with (function, parameter of funtcion)
        if isinstance(hidden_actv_f, (tuple, list)):
            self.hidden_actv_f = hidden_actv_f
        else: 
            self.hidden_actv_f = (hidden_actv_f, 1)

        if isinstance(output_actv_f, (tuple, list)):
            self.output_actv_f = output_actv_f
        else: 
            self.output_actv_f = (output_actv_f, 1)

        self.w_init=w_init # weights in range [-w_init, w_init] for each neu.

        # Parameters for the training
        self.train_MEE = []
        self.train_MSE = []
        self.val_MEE = []
        self.val_MSE = []

        # Counters 
        self.epoch = 0
        self.num_input = 0
        self.num_hidden = 0 

        # Data (and output from hidden) container
        self.transfer_line = None

    # Importing core functions
    from utils_CC._add_hidden_neuron import add_hidden_neuron
    from utils_CC._train import train

    @property
    def out_net(self):
        """Compute the output of the net"""
        return np.array([neu.out(self.transfer_line) for neu in self.out_neurons]).T

    def create_net(self, input_data, n_output):
        """
        Create a new network starting from given data.

        Parameters
        ----------
        input_data: numpy 2d array
            Input data to the net, these will go in the transfer_line.
        n_output: int
            The number of ouput of the network.
        """
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
        """
        Predict the output of given data.
        
        Parameters
        ----------
        data: numpy 2d array
            The data to predict.
        
        Returns
        -------
        numpy 2d array
            Array containing the output of each input data.
        """
        # extract the shape of the data
        n_data, n_features = data.shape
        # Initialize the transfer matrix to empty matrix (orrible but needed)
        self.transfer_line = np.empty((n_data, n_features + self.num_hidden))
        # Fill the first columns of the transfer with the data
        self.transfer_line[:, :n_features] = data
        self.feedforward()# Move the data to the output
        return self.out_net # return the output of the network
