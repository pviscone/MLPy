"""
Module implementing the layer structure.
"""

import numpy as np
from utils.activations import actv_funcs, dactv_funcs

class Layer:
    """
    Layer of the Neural Network.
    """
    def __init__(self, unit_number, input_matrix,val_matrix=[],
                 func = ("sigmoid",1), starting_points = 0.1,
                 preload_w = None, preload_bias = None, from_backup = False):
        """
        __init__ of class Layer.

        Parameters
        ----------
        unit_number : int
            Number of unit in the Layer.
        input_matrix: list or numpy 2d array
            Array or list containing the inputs to the Layer, for example:

            [[feat_1, ... , feat_n],          (data_1)
             [        ...         ]             ...
             [        ...         ]             ...
             [        ...         ]             ...
             [feat_1, ... , feat_n]]          (data_m)

            In wich m is the number of pattern (or input) to the Layer and
            n is the number of features in input (equal to unit_number).
        func : tuple
            Tuple containing the name of the activation function of the Layer
            and the parameter of that function.
            Avaible function are:
                - 'sigmoid'
                - 'linear'
                - 'relu'
            Default choice is 'sigmoid'.
        starting_points : float
            The weight of this Layer will be random inizialized in
            [-starting_points, starting_points]
        preload_w: list
            List with the weights preloaded, if None initialize to random.
        preloaded_bias : list
            List with the bias preloaded, if None initialize to random.
        from_backup : boolean
            If true preload the weights and bias.
        """
        self.input=np.array(input_matrix) # Storing the input matrix
        self.val_data=np.array(val_matrix)
        self.unit_number=unit_number      # Storing the number of units
        self.dW_1=0 #Needed for the momentum in backprop
        self.db_1=0 #Needed for the momentum in backprop
        self.dw_nest=0 #Needed for nestorov momentum
        self.vdw=0 #Needed for RMSProp
        self.vdb=0 #Needed for RMSProp
        num_features = np.shape(self.input)[1] # Number of input features
        '''
        The weight matrix has the structure:

        [[weight_1, ... ,weight_j]       (unit_1)
        [          ...          ]         ...
        [weight_1, ... ,weight_j]]      (unit_s)

        - j is the number of input weights of each unit in the Layer
        - s is the the number of unit in the layer (unit_number).
        '''

        if from_backup:
            self.weight = np.array(preload_w)
        else:
            self.weight=np.random.uniform(-starting_points,starting_points,
                                          size=(unit_number, num_features ) )
        # Array of bias for each unit in the Layer
        if from_backup: self.bias = np.array(preload_bias)
        else:
            self.bias=np.random.uniform(- starting_points, starting_points,
                                        size = unit_number )

        #Storing the activation function and his derivative
        self.function, self.slope=func
        self.func=lambda x : actv_funcs(self.function)(x,self.slope)
        self.der_func=lambda x : dactv_funcs(self.function)(x,self.slope)

    @property
    def net(self):
        """
        This property evaluate the dot product between the inputs and the
        weight (adding the bias).
        """
        return self.input.dot( self.weight.T ) + self.bias

    @property
    def net_nest(self):
        """
        This property evaluate the dot product between the inputs and the
        weight (adding the bias).
        """
        return self.input.dot( (self.weight+self.dw_nest).T ) + self.bias

    @property
    def out(self):
        """
        This property return the output values of the net using the activation
        function.
        """
        #scorrendo le colonne trovi i net di tutti i neuroni, scorrendo le righe cambi pattern
        return self.func(self.net)

    @property
    def net_val(self):
        """
        This property evaluate the dot product between the inputs and the
        weight (adding the bias).
        """
        return self.val_data.dot( self.weight.T ) + self.bias

    @property
    def out_val(self):
        """
        This property return the output values of the net using the activation
        function.
        """
        #scorrendo le colonne trovi i net di tutti i neuroni, scorrendo le righe cambi pattern
        return self.func(self.net_val)

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
