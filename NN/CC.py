"""
Implementation of the Multi Layer Perceptron.
"""
import time
import numpy as np
#from numba import njit # <---- Decomment for a 5x speed up
from utils.losses import MEE_MSE
from utils.activations import actv_funcs, dactv_funcs
from layer import Layer

class Neuron:
    """
    Neuron of the Cascade Correlation Neural Network.
    """
    def __init__(self, input_data, val_data, func = ("sigmoid",1), w_init = 0.1):
        self.input_data = np.array(input_data) # Storing the input matrix
        self.val_data= np.array(val_data)
        self.weight=np.random.uniform(-w_init,w_init, size = len(input_data) )
        # Array of bias for each unit in the Layer
        self.bias=np.random.uniform(- w_init, w_init)

        #Storing the activation function and his derivative
        self.function, self.slope=func
        self.func=lambda x : actv_funcs(self.function)(x,self.slope)
        self.der_func=lambda x : dactv_funcs(self.function)(x,self.slope)

    @property
    def net(self):
        return self.input_data.dot(self.weight) + self.bias
    @property
    def out(self):
        return self.func(self.net)


class CCNN:
    """
    Cascade Correlation Neural Network class.
    """
    def __init__(self, func, w_init, hidden_actv_f, output_actv_f,
                        eta=0.1, lamb=0, norm_L=2, alpha=0, nesterov=False):
        """
        __init__ function of the class.

        Parameters
        ----------
        func : list of tuple
            List that contains:
                [( act. func. name, func. params. ), ..., (...)]
            The number of elements (tuples) of this list is equal to the number
            of Layers of the MLP.
        eta : float
            The learning rate (default is 0.1)
        lamb : float
            The Tikhonov regularization factor (default is 0).
        norm_L : int
            The type of the norm for the Tikhonov regularization (default is 2,
            euclidean).
        alpha : int
            Momentum factor (to do).
        nesterov = False
            Implementing the nesterov momentum (to do).
        starting_point : list or numpy 1d array of float
            The starting weights of each layer (i) is initialized extracting
            from a random uniform distribution in the interval:
                [-starting_point[i], starting_point[i]]
        """
        self.network=[]
        self.input_data = None
        # list of tuple with (function, parameter of funtcion)
        self.hidden_actv_f = hidden_actv_f
        self.output_actv_f = output_actv_f
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


