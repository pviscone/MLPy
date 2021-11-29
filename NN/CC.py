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
    def __init__(self, input_data, func = ("sigmoid",1), w_init = 0.1, connect = True):
        self.input_data = np.array(input_data) # Storing the input matrix
        n_data, n_prev_neurons = input_data.shape
        self.output = np.empty((n_data, n_prev_neurons + 1))
        self.output[:, :-1] = self.input_data
        self.weight=np.random.uniform(-w_init,w_init, size = n_prev_neurons )
        # Array of bias for each unit in the Layer
        self.bias=np.random.uniform( -w_init, w_init )

        #Storing the activation function and his derivative
        self.function, self.slope=func
        self.func=lambda x : actv_funcs(self.function)(x,self.slope)
        self.der_func=lambda x : dactv_funcs(self.function)(x,self.slope)
        self.connect = connect

    @property
    def net(self):
        return self.input_data.dot(self.weight) + self.bias

    @property
    def out(self):
        return self.func(self.net)

    @property
    def out_flow(self):
        if self.connect:
            self.output[:,-1] = self.out
        else: self.output[:,-1] = np.zeros(len(self.output))
        return self.output

class CCNN:
    """
    Cascade Correlation Neural Network class.
    """
    def __init__(self, w_init = 0.1, hidden_actv_f = 'sigmoid', output_actv_f='sigmoid',
                 eta=0.1, lamb=0, norm_L=2, alpha=0, nesterov=False):
        self.hidden=[]
        self.output=[]
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

    @property
    def out_net(self):
        return np.array([neu.out for neu in self.output]).T

    def __getattr__(self,attr):
        """Get the atribute of MLP"""
        return [getattr(neu,attr) for lay in self.network]

    def train(self, input_data, labels, max_epoch, max_hidden = 0, stop_threshold = 0.1):
        if len(self.output) == 0:
            self.create_net(input_data, labels.shape[1])
        self.num_hidden = 0
        add_neuron = False
        while self.num_hidden <= max_hidden:
            if add_neuron:
                print('I will add an hidden')
                self.add_hidden_neuron(input_data)
                self.num_hidden += 1
            for i in range(max_epoch):
                self.feedforward()
                self.output_learning_step(labels)
                MEE, MSE = MEE_MSE(labels, self.out_net)
                self.train_MEE.append(MEE)
                self.train_MSE.append(MSE)
            if MEE < stop_threshold: break
            else: add_neuron = True

    def add_hidden_neuron(self, input_data):
        if self.num_hidden == 0: input_hidden = input_data
        else: input_hidden = self.hidden[-1].out
        self.hidden.append(Neuron(input_hidden, 
                                  func = self.hidden_actv_f, 
                                  w_init = self.w_init, connect = False))
        for out in self.output:
            out.weight = np.append(out.weight, np.random.uniform(-self.w_init, self.w_init))
        return

    def create_net(self, input_data, n_output):
        for i in range(n_output):
            self.output.append(Neuron(input_data, 
                               func = self.output_actv_f, 
                               w_init = self.w_init))

    def feedforward(self):
        """ Move the input to the output of the net"""
        for neu_prev,neu_next in zip(self.hidden[:-1],self.hidden[1:]):
            neu_next.input_data=neu_prev.out_flow
        if len(self.hidden ) > 0:
            for neu_out in self.output:
                neu_out.input_data = self.hidden[-1].out_flow

    def output_learning_step(self, labels):
        for n, neu_out in enumerate(self.output):

            delta=( ( labels[:,n] - neu_out.out ) * neu_out.der_func(neu_out.net) )

            dW = np.sum([i*j for i,j in zip(delta,neu_out.input_data)], axis=0) #batch
            db=np.sum(delta)

            neu_out.weight += self.eta*dW
            neu_out.bias   += self.eta*db

    def hidden_learning_step(self, labels):
        return

    def predict(self, data):
        if len(self.hidden) > 0:
            self.hidden[0].input = data
        else:
            for out in self.output:
                out.input_data = data
        self.feedforward()
        return self.out_net
