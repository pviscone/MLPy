import numpy as np
from utils.activations import actv_funcs, dactv_funcs

class Neuron:
    """
    Neuron of the Cascade Correlation Neural Network.
    """
    def __init__(self, data_shape, func = ("sigmoid",1), w_init = 0.1):
        n_data, n_prev_neurons = data_shape
        self.weight=np.random.uniform(-w_init,w_init, size = n_prev_neurons )
        # Array of bias for each unit in the Layer
        self.bias=np.random.uniform( -w_init, w_init )

        #Storing the activation function and his derivative
        self.function, self.slope=func
        self.func=lambda x : actv_funcs(self.function)(x,self.slope)
        self.der_func=lambda x : dactv_funcs(self.function)(x,self.slope)

    def net(self, input_data):
        return input_data.dot(self.weight) + self.bias

    def out(self, input_data):
        return self.func(self.net(input_data))
