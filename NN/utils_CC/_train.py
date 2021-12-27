"""
Train utilities for the Cascade Correlation Neural Network.
"""

import numpy as np
from neuron_CC import Neuron
from numba import njit
from utils.losses import MEE_MSE

def train(self, input_data, labels, val_data = None, val_labels = None, 
          candidate_RMSProp = False, candidate_beta = 0.5,
          RMSProp = False, beta = 0.5, alpha = 0.,
          eta = 0.1, min_epoch = 10, tot_max_epoch = 1000, max_epoch = 100,
          candidate_lambda = 0., max_hidden = 0, stop_threshold = 0.1, 
          stack_threshold = 0.1, n_candidate = 10, max_candidate_epoch = 100, 
          min_candidate_epoch = None, candidate_eta = None, candidate_factor = 10):
    """
    (Function of the class)
    Train function for the Cascade Correlation.

    Parameters
    ----------
    input_data : numpy 2d array
        Training data.
    labels : numpy 2d array
        Labels of the training data.
    eta : float
        Learning rate for output neurons. Default is 0.1.
    min_epoch : int
        Minimum number of epoch for the training of the output neuron before
        to start thinking about add a new unit.
    max_epoch : int
        Maximum nuber of epoch before stop the (global) training algorithm.
    max_hidden : int
        Maximum number of hidden neurons.
    stop_threshold : float
        Error value for wich the task is considered solved (and the training 
        algorithm stops).
    stack_threshold : float
        Consider the training of the network with a certain number of hidden
        neuron, lets consider just the training error and the number of epoch
        (n_epoch) of this portion of training algorithm.
        - Define the previous training error minus the last training error: e.
        - Define the first training error minus the last training error: E.
        - Define the relative error [(e/E) * n_epoch]: r
        Now if r is lower than stack_threshold the training of this network 
        configuration stop and the algorithm add an hidden neuron.
        The idea behind this stop condion is based on the ratio between the 
        global derivative of the error and the local deivative of the error.
    n_candidate : int
        Number of candidate neurons tested before add a new neuron to the net.
    max_candidate_epoch : int
        Maximum epoch for the training of the hidden neurons.
    min_candidate_epoch : int
        Minimum epoch for the training of the hidden neurons, if None the 
        algorithm will choose automatically 
            min_candidate_epoch = 0.1*max_candidate_epoch
    candidate_eta : float
        Learning rate for the learning of hidden neurons, if None it will be 
        used the same value of eta.
    """
    self.eta = eta # Create the learning rate
    n_data, n_features = input_data.shape
    if isinstance(val_data, np.ndarray): check_val = True
    else: check_val = False
    if self.num_input == 0: # If the network is empty create a new net
        self.create_net(input_data, labels.shape[1]) # Create new net
        self.num_hidden = 0 # Initialize zero hidden
    else: # Otherwise just fill the input in the transfer line
        self.transfer_line = np.ones((n_data, n_features + self.num_hidden))
        self.transfer_line[:, :n_features] = input_data

    add_neuron = True # Initialize boolean that decides if add a new neuron
    log_string = ''
    
    # While the hidden neuron are less then the maximum given by user
    # or the train converge before of max_hidden
    while add_neuron:
        epoch = 0 # Set the epoch to zero
        keep_train = True # Keep train parameter for output training
        start_epoch = self.epoch # Epoch of starting training
        while keep_train: # 
            # TRAIN OUT PHASE #################################################
            output_learning_step(labels, self.out_neurons, 
                                 self.transfer_line, self.eta,
                                 RMSProp, beta, alpha) # Learning step
            MEE, MSE = MEE_MSE(labels, self.out_net) # Compute train errors
            print(f'train [{self.num_hidden} hidd][MEE={MEE:.2f}]      ', end = '\r')
            # CHECK TRAIN STATUS PHASE ########################################
            if epoch > max(min_epoch, 1): # If epoch reach min epoch
                # Compute the variation of error from start training to now
                tot_delta_E = np.abs(self.train_MSE[start_epoch]-MSE)
                # Compute the last variation of error
                last_delta_E = np.abs(self.train_MSE[-1]-MSE)
                # Compute the (relative delta error) * epoch
                relative_delta_E = (last_delta_E/tot_delta_E) * epoch
                if relative_delta_E < stack_threshold: # Stop condition
                    keep_train = False 
            self.train_MEE.append(MEE) # append errors
            self.train_MSE.append(MSE)
            if check_val:
                MEE, MSE = MEE_MSE(val_labels, self.predict(val_data))
                self.val_MEE.append(MEE)
                self.val_MSE.append(MSE)
                self.transfer_line = np.ones((n_data, n_features + self.num_hidden))
                self.transfer_line[:, :n_features] = input_data
                self.feedforward()
            epoch += 1 # update local epochs
            self.epoch+=1 # update global epochs
            if MSE < stop_threshold: # If error required reached stop train
                add_neuron = False
                keep_train = False
            if self.epoch > tot_max_epoch: # If max_epoch reached stop training
                add_neuron = False
                keep_train = False
            if epoch > max_epoch:
                keep_train = False
                add_neuron = True
        # ADD NEURON PHASE #####################################################
        if self.num_hidden >= max_hidden: 
            add_neuron = False# Get out if max_hidden reached
        if add_neuron: # If add a Neuron is True add the neuron
            self.add_hidden_neuron(labels, n_candidate = n_candidate, 
                                   candidate_eta = candidate_eta,
                                   max_epoch = max_candidate_epoch,
                                   min_epoch = min_candidate_epoch, 
                                   candidate_lambda = candidate_lambda,
                                   candidate_factor = candidate_factor, 
                                   RMSProp = candidate_RMSProp, 
                                   beta = candidate_beta)
        ########################################################################
    print(log_string) # Print the log info about the training



def output_learning_step(labels, out_neurons, transfer_line, eta, RMSProp, beta, alpha):
    """
    Learning step for the Output Neurons.
    """
    # Each out Neuron is independent from others
    for n, neu_out in enumerate(out_neurons): # Loop on output neurons 
        # Compute the out of this output neuron
        out_from_neu = neu_out.out(transfer_line)
        # Compute the net of this output neuron
        net = neu_out.net(transfer_line)
        # Conpute the derivative of the net of this output neuron
        dnet = neu_out.der_func(net)
        # Compute the gradient of this output neuron (see the jit function)
        grad_W, grad_b = out_jit_gradient(out_from_neu, net, dnet, 
                                          n, transfer_line, labels)
        if RMSProp:
            dW, db = neu_out.RMSProp(grad_W, grad_b, eta, beta)
        else:
            dW = eta*grad_W
            db = eta*grad_b
        # Update the weights (and bias)
        neu_out.weight += dW + alpha * neu_out.old_dW
        neu_out.bias   += db + alpha * neu_out.old_db
        neu_out.old_dW = dW
        neu_out.old_db = db

@njit(cache = True, fastmath = True)
def out_jit_gradient(out_from_neu, net, dnet, n, inputs, labels):
    """
    This function compute the gradient of the Error with a batch approach.
    """
    # Compute the delta for the out neuron (like first step of BP)
    delta=( ( labels[:,n] - out_from_neu) * dnet )
    list_prod = np.empty(inputs.shape) # Initialize an array for delta*input
    # Computing for each input the quantity delta * input
    for i in range(len(inputs)): # speed up this loop on data with numba!
        list_prod[i] = delta[i] * inputs[i] 
    # Compute the grad_w as the sum of the (delta * input) values for each input 
    grad_w = np.sum(list_prod, axis = 0) 
    # Compute the grad_b as the sum of the delta
    grad_b = np.sum(delta)
    return grad_w, grad_b
