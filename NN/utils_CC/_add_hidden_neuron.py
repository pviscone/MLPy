import numpy as np
from neuron_CC import Neuron
import matplotlib.pyplot as plt
from numba import njit

def add_hidden_neuron(self, labels, n_candidate = 50, candidate_eta = None,
                      RMSProp = False, beta = 0.5,
                      candidate_stack_threshold = 0.05, patience = 10, 
                      max_epoch = 500, min_epoch = None, candidate_lambda = 0.,
                      candidate_factor = 10):
    """
    Function that manage the addition of an hidden neuron in the net.
    The neuron is added directy in self.hidden, also the array of weights of 
    the output is updated (with a new random value).

    Parameters
    ----------
    labels: numpy 2d array
        Array with the label (train output) of the data.
    n_candidate : int
        Number of candidate to train.
    candidate_eta : float
        Learning parameter for each candidate.
    candidate_stack_threshold : float
        The treshold for considering the S function stacked.
    patience : int
        If a candidate's S function is stacked for more than patience time 
        than stop the training.
    max_epoch : int
        Maximum number of epoch for the train of each candidate.
    min_epoch : int
        Minimum number of epoch to train, if None this is fixed to max_epoch/10
    """
    # PREPARATION PHASE #######################################################
    # (Define variables and function for the solution)

    # If the candidate_eta is None just use the same eta of the output training
    if candidate_eta == None: candidate_eta = self.eta # (could be not good...)
    # Save the shape of the transfer_line (the shape of the data + out_hidden)
    transfer_data_shape = self.transfer_line.shape

    ## Quantity that needs to be compute just one time ##
    # Save the output(s) of the network
    out_net = self.out_net
    # Error: difference between the outputs and the labels
    E = out_net - labels
    # Delta Error: residual of the error E
    delta_E = E - np.mean(E, axis = 0)

    def generate_neuron(i):
        """Generate a basic hidden neuron"""
        return Neuron(data_shape = transfer_data_shape,
                      func = self.hidden_actv_f, 
                      w_init = self.w_init*i)

    def compute_S(labels, candidate):
        """
        Compute the correlation function S.

        Parameters
        ----------
        labels: numpy 2d array
            Array with the label (train output) of the data.
        candidate : Neuron
            Candidate neuron.
        Returns
        -------
        float
            The value of S.
        """
        S = 0 # S will be a sum, so start from zero
        o_p = candidate.out(self.transfer_line) # Compute the candidate output
        for k in range(self.num_output): # For each output (of the net)
            E_pk = out_net[:,k] - labels[:,k] # extract the relative error
            delta_E = E_pk - np.mean(E_pk) # Compute the residual for that out
            # Compute the quantity S_k: S for that output
            S_k = np.abs(np.sum((o_p - np.mean(out_net))*delta_E))
            S += S_k # Sum all the S_k gives S
        return S

    def grad_desc_step(labels, candidate):
        """Training step with gradient descend
        Parameters
        ----------
        labels: numpy 2d array
            Array with the label (train output) of the data.
        candidate : Neuron
            Candidate neuron.
        """
        # Compute the gradient
        w_grad, b_grad = gradient(labels, candidate, delta_E, out_net)
        if RMSProp:
            dW,db=candidate.RMSProp(w_grad, b_grad, candidate_eta, beta)
#            dW = -dW
#            db = -db
        else: 
            dW = candidate_eta*w_grad
            db = candidate_eta*b_grad
        reg_w = candidate_lambda * np.abs(candidate.weight) # Regularizat. term
        reg_b = candidate_lambda * np.abs(candidate.bias) # Regularizat. term
        candidate.weight += dW - reg_w # update weights
        candidate.bias   += db - reg_b # update bias
    
    def gradient(labels, candidate, delta_E, out_net):
        """
        Wrapper function that call the external jitted function to compute the 
        gradient. This is a wrapper because extract all the values from the
        network need to compute the gradient and pass them to the compiled 
        function below.

        Parameters
        ----------
        labels: numpy 2d array
            Array with the label (train output) of the data.
        candidate : Neuron
            Candidate neuron.
        delta_E: float
            Residual error on output defined as:
            ----> delta_E = E - np.mean(E, axis = 0)
            where E = self.out_net - labels
        out_net : float
            The output of the network (self.out_net)
        
        Returns
        -------
        (numpy 1d array of float, float)
            (Gradient of the weights, gradient of the bias)
        """
        # Extract the needed values
        o_p = candidate.out(self.transfer_line)
        dnet = candidate.der_func(candidate.net(self.transfer_line))
        weight = candidate.weight
        params = [self.transfer_line, self.num_output, o_p, dnet,
                  out_net, delta_E, weight, labels]
        # Call the compiled function
        return jit_gradient(*params)
    ############################################################################

    # ALGORITHM PHASE ##########################################################
    # Solving the task using the previous definitions

    # Start from very negative S_best
    best_S = -np.infty 
    # Loop over the candidate
    for i in range(n_candidate):
        # Generate a neuron and compute the initial S
        candidate = generate_neuron(candidate_factor)
        S_start = compute_S(labels, candidate)
        # Initialize the epochs and the patience
        epoch = 0
        count_patience = 0
        if min_epoch == None: min_epoch = int(max_epoch/10) # Arbitary
        for _ in range(min_epoch): # first train for min_epoch epochs
            grad_desc_step(labels, candidate)
            S = compute_S(labels, candidate)
            candidate.S_list.append(S) # Update the list of S of the candidate

        epoch = min_epoch # update the number of epoch
        S_prev = S # Just initialize S_prev

        # Loop checking the two stopping parameters
        while (epoch < max_epoch) and (count_patience < patience):
            grad_desc_step(labels, candidate) # Update the weights
            S = compute_S(labels, candidate)  # Compute S
            print(f'hidd {self.num_hidden + 1}:[{i}/{n_candidate}][S={best_S:.0f}][MEE:{self.train_MEE[-1]:.2f}]   ', end='\r')
            # Evaluate the difference from the S_start
            total_S_diff = np.abs(S_start - S) 
            # Relative difference: compute how much I could move if I 
            # make epoch times just the last movement (S-S_prev) and 
            # compare it with the total movement.
            rel_S_diff = np.abs(epoch * (S - S_prev)/total_S_diff)
            # If relative movement is lower than the treshold update patience
            if rel_S_diff < candidate_stack_threshold:
                count_patience += 1
            S_prev = S # Update S_prev
            candidate.S_list.append(S) # Append to the list of S the last value
            epoch += 1

        if S > best_S: # Compare the last candidate with the best one
            # If the score (S) is greater than the prev best
            # update the best candidate
            best_S = S
            best_neu = candidate
            best_neu.epoch_trained = epoch
 
    self.hid_neurons.append(best_neu) # Add to the net the best candidate
    # Compute the out of this new neuron
    out_new_neu = self.hid_neurons[-1].out(self.transfer_line)

    # Update the transfer line adding the output of the neuron
    # NOTE: this will never change anymore!!
    self.transfer_line = np.column_stack((self.transfer_line, out_new_neu))
    # Update the output weights adding a new random value
    for out in self.out_neurons:
        out.weight = np.append(out.weight, np.random.uniform(-self.w_init, self.w_init))
        out.vdw = np.append(out.vdw, 0)
        out.old_dW = np.append(out.old_dW, 0)
    # Update the number of hidden neurons
    self.num_hidden += 1

@njit(cache = True, fastmath = True)
def jit_gradient(transfer_line, num_output, o_p, dnet, out_net, delta_E, weight, labels):
    """
    Gradient of S in a jit function.

    Parameters
    ----------
    transfer_line : numpy 2d array
        Array containing all the data and the output of the other input.
    num_output : int
        Number of output of the network.
    o_p : numpy 1d array
        Output of the candidate neuron.
    dnet : numpy 1d array
        Derivative of the activation function applied to the net of the
        candidate neuron.
    out_net : numpy 2d array
        Output of the total network.
    delta_E : numpy 2d array
        Residual error of the network.
    weight : numpy 1d array
        Weight of the candidate neuron.
    labels : numpy 2d array
        Output labels for training.
    
    Returns
    -------
    (numpy 1d array of float, float)
        (gradient for the weights, gradient for the bias).
    """
    # Define an array of zeros for the gradient
    w_grad = np.zeros(weight.shape)
    b_grad = 0 # same for the bias
    for k in range(num_output): # loop over the outputs
        # For each output we compute S_k with the usual formula
        S_k = np.sum((o_p - np.mean(out_net))*delta_E[:,k])
        # The delta uses the sign of S_k
        delta = np.sign(S_k) * delta_E[:,k] * dnet
        # For each input we compute the delta
        delta_per_input = np.empty(transfer_line.shape)
        for i in range(len(delta)): # len(delta)=len(transfer_line)
            delta_per_input[i] = delta[i]*transfer_line[i] # usual delta
        w_grad += np.sum(delta_per_input, axis=0) # gradient as sum of delta
        b_grad += np.sum(delta)
    return w_grad, b_grad # return the gradients
