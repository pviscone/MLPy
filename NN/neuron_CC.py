import numpy as np
from utils.activations import actv_funcs, dactv_funcs

class Neuron:
    """
    Neuron of the Cascade Correlation Neural Network.
    """
    def __init__(self, data_shape, func = ("sigmoid",1), w_init = 0.1):
        """
        __init__ function of the Neuron.

        Parameters
        ----------
        data_shape : tuple or list
            Tuple or list contaning the data shape coming from the inputs 
            or from both inputs and the previous hidden.
        func : tuple
            Tuple containing a string with the activation function name
                - sigmoid
                - linear
                - relu
            and the parameter of the relative function (decay for the sigmoid, 
            angular coefficient for the linear/relu).
        w_init : float
            the weights will be sampled from a random uniform distribution in 
            [-w_init, w_init], also the bias will be extracted in this interval.
        """
        # Separate number of data and number of previous neuron (considering 
        # inputs as neurons)
        n_data, n_prev_neurons = data_shape

        # Initialize the weights
        self.weight=np.random.uniform(-w_init,w_init, size = n_prev_neurons )
        # Initialize the bias (just a number...)
        self.bias=np.random.uniform( -w_init, w_init )

        #Storing the activation function and his derivative
        self.function, self.slope=func
        self.func=lambda x : actv_funcs(self.function)(x,self.slope)
        self.der_func=lambda x : dactv_funcs(self.function)(x,self.slope)

        # Initialize two attribute for the history of hidden unit during 
        # their training
        self.S_list = []
        self.epoch_trained = 0
        self.vdw = np.zeros(n_prev_neurons)
        self.vdb = 0
        self.old_dW = np.zeros(n_prev_neurons)
        self.old_db = 0

    def net(self, input_data):
        """Net function for the neuron"""
        return input_data.dot(self.weight) + self.bias

    def out(self, input_data):
        """Out function for the neuron"""
        return self.func(self.net(input_data))

    def RMSProp(self, grad_W, grad_b, eta, beta):
        """
        Update the weights and bias of the Layer using RMSProp.
        """
        # Update the weights
        self.vdw = beta*self.vdw + (1-beta)*grad_W**2
        self.vdb = beta*self.vdb + (1-beta)*grad_b**2
        dW = eta*grad_W/(np.sqrt(self.vdw) + 1e-8)
        db = eta*grad_b/(np.sqrt(self.vdb) + 1e-8)
        return dW, db
