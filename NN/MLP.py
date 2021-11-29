"""
Implementation of the Multi Layer Perceptron.
"""
import time
import os
from pathlib import Path
import json
import numpy as np
#from numba import njit # <---- Decomment for a 5x speed up
from utils.losses import MEE_MSE
from layer import Layer

class MLP:
    """
    Multi Layer Perceptron class.
    """
    def __init__(self, structure = [], func = None, starting_points = 0.1,
                 eta=0.1, lamb=0, norm_L=2, alpha=0, nesterov=False, 
                 filename = None, epoch_to_restore = -1):
        """
        __init__ function of the class.

        Parameters
        ----------
        structure : list
            List containing the number of unit of each Layer in the MLP.
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
        filename : string
            Restore the net from a json file 'filename', if None initialize a
            new net.
        epoch_to_restore : int
            Number of epoch to restore (if want to reload a pretrained net).
            If -1 reload the network fully trained.
        """
        self.filename = filename
        self.network=[]
        if self.filename == None: # If the user doesn't want to restore a previous net
            self.structure=structure # Number of units per layer
            # list of tuple with (function, parameter of funtcion)
            self.func=[f if isinstance(f, (tuple, list)) else (f, 1) for f in func]
            self.starting_points=starting_points # start_point list for random weights
            self.eta = eta # Learning rate
            self.lamb = lamb # Tikhonov regularization
            self.norm_L = norm_L # Tikhonov regularization norm
            self.alpha = alpha # Parameter for momentum
            self.nesterov = nesterov # If the user want the nesterov momentum
            self.epoch = 0 # set number of epoch to 0
            self.train_MEE = []
            self.train_MSE = []
            self.val_MEE = []
            self.val_MSE = []
        else: # If the user want to load a pretrained model
            with open(self.filename) as json_file: # open the file with the json net
                data = json.load(json_file) # Load the file in a dictionary
                for key, val in data.items(): # Create all the attribute from file
                    if ('weights' not in key) and ('bias' not in key):
                        setattr(self, key, val)
            # If epoch_to_restore = -1 restore the last epoch
            if epoch_to_restore == -1: self.epoch_to_restore = self.epoch
            else: self.epoch_to_restore = epoch_to_restore
            # Create a net with empty input/val just to be able to predict without training
            self.create_net(np.empty(data['input_data_shape']), np.empty(data['val_data_shape']))

    def __getattr__(self,attr):
        """Get the atribute of MLP"""
        return [getattr(lay,attr) for lay in self.network]

    def train(self, input_data, labels, val_data, val_labels, epoch,
              clean_net = False, save_rate = -1, filename = None):
        """
        Parameters
        ----------
        input_data : list or numpy 2d array
            Array with the data in input to the MLP.
        labels : list or numpy 2d array
            Labels of the input_data.
        epoch : int
            Number of epoch for training.
        val_data : list or numpy 2d array
            Array with the validation data in input to the MLP.
        val_labels : list or numpy 2d array
            Labels of the val_data.
        clean_net : boolean
            If True restore the net, if False keep the pretrained net (if exist)
        save_rate : int
            Rate of saving of the net during the training
        """
        # Check if all the input are numpy arrays
        input_data = np.array(input_data)
        labels = np.array(labels)
        val_data = np.array(val_data)
        val_labels = np.array(val_labels)

        # Reset the net if clean_net == True
        if clean_net:
            self.train_MEE = []
            self.train_MSE = []
            self.val_MEE = []
            self.val_MSE = []
            self.epoch = 0

        # If the net is just an empty list fill it with the layers
        if len(self.network) == 0:
            self.create_net(input_data, val_data)

        if self.filename != None: # If is a preloaded net fill the input
            self.network[0].input = input_data
            self.network[0].val_data = val_data

        # Start train the net
        total_time = 0
        real_start = time.time()
        print(f'Starting training {self.epoch} epoch', end = '\r')
        for i in range(epoch):
            start_loop = time.time()

            # Train dataset #
            self.feedforward()
            self.learning_step(labels)
            MEE, MSE = MEE_MSE(labels,self.network[-1].out)
            self.train_MEE.append(MEE)
            self.train_MSE.append(MSE)

            # Validation dataset #
            self.feedforward_validation()
            MEE, MSE = MEE_MSE(val_labels, self.network[-1].out_val)
            self.val_MEE.append(MEE)
            self.val_MSE.append(MSE)

            # Printing the error
            string_val_err = f'  [val MEE = {self.val_MEE[-1]:.4f}]'
            string_err = f'  [train MEE = {self.train_MEE[-1]:.4f}]'
            string_err += string_val_err

            # Printing remaining time
            elapsed_for_this_loop = time.time()-start_loop
            total_time += elapsed_for_this_loop
            mean_for_loop = total_time/(i+1)
            remain_time = mean_for_loop*(epoch-i)
            string_time = f'  [wait {remain_time:.1f} s]'
            print(f'[Epoch {self.epoch}]' + string_err + string_time + ' '*10, end = '\r', flush = True)

            if i == save_rate:
                self.save_network(filename)

            # Updating epoch
            self.epoch += 1

        # Final print
        print(f'Epoch {self.epoch}:' + string_err + ' '*30, end = '\n')
        print(f'Elapsed time: {time.time()-real_start} s')

    def predict(self, data):
        """
        Predict out for new data

        Parameters
        ----------
        data : list or numpy 2d array
            Array with the data in input to the MLP.

        Returns
        -------
        numpy 1d array or numpy 2d array
            Array with the output labels predicted by the model.
        """
        self.network[0].input = data
        self.feedforward()
        return self.network[-1].out

    def create_net(self, input_data, val_data, w_list = [], bias_list = []):
        """
        Feed the input of the net and propagate it

        Parameters
        ----------
        input_data : list or numpy 2d array
            Array with the data in input to the MLP.
        """
        if self.filename == None: 
            from_backup = False
            w, bias = [None]*len(self.structure), [None]*len(self.structure)
        else:
            from_backup = True
            with open(self.filename, 'r') as json_file:
                data = json.load(json_file)
                w = data[f'weights_epoch{self.epoch_to_restore}']
                bias = data[f'bias_epoch{self.epoch_to_restore}']
        for layer,num_unit in enumerate(self.structure):
            if layer==0: 
                self.network.append(Layer(num_unit,input_data,
                                    val_matrix=val_data,
                                    func=self.func[layer],
                                    preload_w = w[layer],
                                    preload_bias = bias[layer], 
                                    from_backup = from_backup))
            else:
                self.network.append(Layer(num_unit,self.network[layer-1].out, 
                                    val_matrix=self.network[layer-1].out_val,
                                    func=self.func[layer],
                                    preload_w = w[layer],
                                    preload_bias = bias[layer],
                                    from_backup = from_backup))


    def feedforward(self):
        """ Move the input to the output of the net"""
        for lay_prev,lay_next in zip(self.network[:-1:],self.network[1::]):
            lay_next.input=lay_prev.out

    def feedforward_validation(self):
        """ Move the input to the output of the net"""
        for lay_prev,lay_next in zip(self.network[:-1:],self.network[1::]):
            lay_next.val_data=lay_prev.out_val

    def learning_step(self,labels):
        """
        Implementing the backpropagation.

        Parameters
        ----------
        labels : list or numpy 2d array
            Labels of the input_data.
        """
        for reverse_layer_number,layer in enumerate(self.network[::-1]):
            if reverse_layer_number==0:
                delta=((labels-layer.out)*layer.der_func(layer.net))
            else:
                delta=(np.matmul(delta,weight_1)*layer.der_func(layer.net))
            weight_1=layer.weight

#xxxxxxxxxxx Comment here for  5x speed up xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
            dW=np.sum([np.outer(i,j) for i,j in zip(delta,layer.input)], axis=0) #batch
            db=np.sum(delta,axis=0)
#xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

#%%%%%%%%%%% Decomment here for 5x speed up %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#            dW, db = self._jit_update_weight(delta, layer.input)
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

            layer.weight += self.eta*dW - self.lamb * (np.abs(layer.weight)**(self.norm_L-1))*np.sign(layer.weight)
            layer.bias   += self.eta*db

#%%%% Decomment the block for a 5x speed up %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#    @staticmethod
#    @njit(cache = True, fastmath = True)
#    def _jit_update_weight(delta, inputs):
#        list_prod = np.empty( (*delta.shape, inputs.shape[1]) )
#        for i in range(len(delta)): # speed up this loop on data with numba!
#            list_prod[i] = np.outer(delta[i], inputs[i])
#        dW = np.sum(list_prod, axis = 0)
#        db = np.sum(delta, axis = 0)
#        return dW, db
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


    def save_network(self, filename):
        """
        Function that save the net in a json file

        Parameters
        ----------
        filename : string
            The output filename.
        """
        # Write general info of the net
        path_file = Path(filename) # Create a path object
        if not os.path.exists(path_file.parent): # If the folder not exist create it
            os.makedirs(path_file.parent)
        if not path_file.is_file(): # If the file doesn't exist initialize 
#                                   # the net parameters
            net_dict = self.__dict__ # Dict of the MLP object
            # Stuff that we don't write to the output file
            skip_list = ['network', 'filename']
            # (network will be written below)
            save_dict = {} # initialize the dict to write
            save_dict['input_data_shape'] = self.network[0].input.shape
            save_dict['val_data_shape'] = self.network[0].val_data.shape
            for net_attr in net_dict.keys():
                if net_attr not in skip_list: # Write the values in the dict
                    save_dict[net_attr] = net_dict[net_attr]
        else: # If the file exist (like when we save during the training)
            with open(filename, 'r') as json_file:
                save_dict = json.load(json_file) # Just load the json and go on

        # Saving the weights and the bias for each layer
        weights_list = []
        bias_list = []
        for i, layer in enumerate(self.network):
            weights_list.append(layer.weight.tolist())
            bias_list.append(layer.bias.tolist())
        save_dict[f'weights_epoch{self.epoch}'] = weights_list
        save_dict[f'bias_epoch{self.epoch}'] = bias_list

        # Save the output file
        with open(filename, 'w') as outfile:
            json.dump(save_dict, outfile)
