"""
Train utilities for the Cascade Correlation Neural Network.
"""

import numpy as np
from neuron_CC import Neuron
from numba import njit
from utils.losses import MEE_MSE

def train(self, input_data, labels, min_epoch = 10, max_epoch = 1000, 
          max_hidden = 0, stop_threshold = 0.1, stack_threshold = 0.1,
          n_candidate = 10, max_candidate_epoch = 100):
    """
    Train function for the Cascade Correlation.
    """
    if self.num_input == 0: # If the network is empty create a new net
        self.create_net(input_data, labels.shape[1]) # Create new net
        self.num_hidden = 0 # Initialize zero hidden
    else: # Otherwise just fill the input in the transfer line
        self.transfer_line[:,:input_data.shape[1]] = input_data

    add_neuron = True # Initialize boolean that decides if add a new neuron
    log_string = ''
    
    # While the hidden neuron are less then the maximum given by user
    # or the train converge before of max_hidden
    while add_neuron and (self.num_hidden <= max_hidden):  
        epoch = 0 # Set the epoch to zero
        keep_train = True # Keep train parameter for output training
        start_epoch = self.epoch # Epoch of starting training
        while keep_train: # 
            self.feedforward() # Update the transfer line
            self.output_learning_step(labels) # Learning step
            MEE, MSE = MEE_MSE(labels, self.out_net) # Compute train errors
            if epoch > max(min_epoch, 1): # If epoch reach min epoch
                # Compute the delta error from start training to now
                tot_delta_E = np.abs(self.train_MSE[start_epoch]-MSE)
                # Compute the last delta error
                last_delta_E = np.abs(self.train_MSE[-1]-MSE)
                # Compute the relative delta error * epoch
                relative_delta_E = last_delta_E*epoch/tot_delta_E
                if relative_delta_E < stack_threshold: # Stop condition
                    keep_train = False 
                    log_string += f'Train with {self.num_hidden} hidden' +\
                                   ' ended cause error stacked\n'
            self.train_MEE.append(MEE) # append errors
            self.train_MSE.append(MSE)
            epoch += 1 # update local epochs
            self.epoch+=1 # update global epochs
            if MSE < stop_threshold: # If error required reached stop train
                log_string +=f'Train with {self.num_hidden} hidden neu' +\
                              ' ended because error low enough\n'
                add_neuron = False
                keep_train = False
            if self.epoch > max_epoch: # If max_epoch reached stop training
                log_string +=f'Train with {self.num_hidden} hidden neu' +\
                               ' ended because max epoch reached\n'
                add_neuron = False
                keep_train = False
        if self.num_hidden == max_hidden: 
            add_neuron = False# Get out if max_hidden reached
        if add_neuron: # If add a Neuron is True add the neuron
            print('adding an hidden...', end='\r')
            self.add_hidden_neuron(labels, n_candidate = n_candidate, 
                                   max_epoch = max_candidate_epoch)
            print('adding an hidden -> Hidden added, training the new model.')
    print(log_string)



def output_learning_step(self, labels):
    for n, neu_out in enumerate(self.out_neurons):

        out_from_neu = neu_out.out(self.transfer_line)
        net = neu_out.net(self.transfer_line)

        delta=( ( labels[:,n] - out_from_neu) * neu_out.der_func(net) )

        dW = np.sum([i*j for i,j in zip(delta, self.transfer_line)], axis=0) #batch
        db = np.sum(delta)

        neu_out.weight += self.eta*dW
        neu_out.bias   += self.eta*db
