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
    def __init__(self, data_shape, func = ("sigmoid",1), w_init = 0.1, connect = True):
        n_data, n_prev_neurons = data_shape
        self.weight=np.random.uniform(-w_init,w_init, size = n_prev_neurons )
        # Array of bias for each unit in the Layer
        self.bias=np.random.uniform( -w_init, w_init )

        #Storing the activation function and his derivative
        self.function, self.slope=func
        self.func=lambda x : actv_funcs(self.function)(x,self.slope)
        self.der_func=lambda x : dactv_funcs(self.function)(x,self.slope)
        self.connect = connect

    def net(self, input_data):
        return input_data.dot(self.weight) + self.bias

    def out(self, input_data):
        return self.func(self.net(input_data))

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

    @property
    def out_net(self):
        return np.array([neu.out(self.transfer_line) for neu in self.out_neurons]).T

    def train(self, input_data, labels, min_epoch = 10, max_hidden = 0, 
              stop_threshold = 0.1, stack_threshold = 0.1,
              n_candidate = 10, candidate_epoch = 100):
        if self.num_input == 0:
            self.create_net(input_data, labels.shape[1])
        self.num_hidden = 0
        add_neuron = False
        while self.num_hidden <= max_hidden:
            epoch = 0
            relative_delta_E = 2*stop_threshold
            sum_delta_E = 0
            if add_neuron:
                print('I will add an hidden')
                self.add_hidden_neuron(labels, n_candidate = n_candidate, candidate_epoch = candidate_epoch)
            while (relative_delta_E > stack_threshold):# and epoch < max_epoch):
                self.feedforward()
                self.output_learning_step(labels)
                MEE, MSE = MEE_MSE(labels, self.out_net)
                if epoch > 0:
                    delta_E = self.train_MSE[-1] - MSE
                    sum_delta_E += delta_E
                if epoch > min_epoch:
                    mean_delta_E = sum_delta_E/epoch
                    relative_delta_E = delta_E/mean_delta_E
                self.train_MEE.append(MEE)
                self.train_MSE.append(MSE)
                epoch += 1
            if MSE < stop_threshold: break
            else: add_neuron = True

    def add_hidden_neuron(self, labels, n_candidate = 50, candidate_epoch = 100):
        def generate_neuron():
            return Neuron(data_shape = self.transfer_line.shape,
                          func = self.hidden_actv_f, 
                          w_init = self.w_init, connect = False)
        def compute_S(labels, candidate):
            S = 0
            for k in range(self.num_output):
                o_p = candidate.out(self.transfer_line)
                E_pk = self.out_net[:,k] - labels[:,k]
                meanE_pk = np.mean(E_pk)
                delta_E = E_pk - meanE_pk
                S_k = np.sum((o_p - np.mean(self.out_net))*delta_E)
                S += S_k
            return S

        candidate = generate_neuron()
        best_neu = candidate
        best_S = 0
        for i in range(n_candidate):
            for j in range(candidate_epoch):
                self.feedforward()
                self.hidden_learning_step(labels, candidate)
            S = compute_S(labels, candidate)
            if S > best_S:
                best_S = S
                best_neu = candidate
            candidate = generate_neuron()

        self.hid_neurons.append(best_neu)

        out_new_neu = self.hid_neurons[-1].out(self.transfer_line)

        n_data = len(self.transfer_line)
        self.transfer_line = np.column_stack((self.transfer_line, out_new_neu))
        for out in self.out_neurons:
            out.weight = np.append(out.weight, np.random.uniform(-self.w_init, self.w_init))
        self.num_hidden += 1

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

    def output_learning_step(self, labels):
        for n, neu_out in enumerate(self.out_neurons):

            out_from_neu = neu_out.out(self.transfer_line)
            net = neu_out.net(self.transfer_line)

            delta=( ( labels[:,n] - out_from_neu) * neu_out.der_func(net) )

            dW = np.sum([i*j for i,j in zip(delta, self.transfer_line)], axis=0) #batch
            db = np.sum(delta)

            neu_out.weight += self.eta*dW
            neu_out.bias   += self.eta*db

    def hidden_learning_step(self, labels, candidate):
        o_p = candidate.out(self.transfer_line)
        dnet = candidate.der_func(candidate.net(self.transfer_line))
        dW = np.zeros(len(candidate.weight))
        for k in range(self.num_output):
            E_pk = self.out_net[:,k] - labels[:,k]
            meanE_pk = np.mean(E_pk)
            delta_E = E_pk - meanE_pk
            S_k = np.sum((o_p - np.mean(self.out_net))*delta_E)
            delta = np.sign(S_k) * delta_E * dnet
            dW = np.sum([i*j for i,j in zip(delta, self.transfer_line)], axis=0)
            db = np.sum(delta) 
            candidate.weight += self.eta * dW
            candidate.bias += self.eta * db

    def predict(self, data):
        n_data, n_features = data.shape
        self.transfer_line = np.ones((n_data, n_features + self.num_hidden))
        print(self.transfer_line[:, :n_features].shape)
        self.transfer_line[:, :n_features] = data
        self.feedforward()
        return self.out_net
