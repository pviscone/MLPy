"""
Implementation of the Multi Layer Perceptron.
"""
import time
import os
from pathlib import Path
import json
import numpy as np
from numba import njit
from utils.losses import MEE_MSE
from layer import Layer


class MLP:
    """
    Multi Layer Perceptron class.
    """
    def __init__(self, structure = [], func = None, starting_points=None,
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
            self.preloaded = False
            self.structure=structure # Number of units per layer
            # list of tuple with (function, parameter of funtcion)
            self.func=[f if isinstance(f, (tuple, list)) else (f, 1) for f in func]
            if starting_points == None: self.starting_points = [0.1]*len(self.structure)
            else: self.starting_points=starting_points # start_point list for random weights
            self.train_MEE = []
            self.train_MSE = []
            self.val_MEE = []
            self.val_MSE = []
            self.train_accuracy=[]
            self.val_accuracy=[]
            self.epoch = 0 # set number of epoch to 0
        else: # If the user want to load a pretrained model
            self.preloaded = True
            with open(self.filename) as json_file: # open the file with the json net
                data = json.load(json_file) # Load the file in a dictionary
                for key, val in data.items(): # Create all the attribute from file
                    if ('weights' not in key) and ('bias' not in key):
                        setattr(self, key, val)
            # If epoch_to_restore = -1 restore the last epoch
            if epoch_to_restore != -1:
                self.epoch = epoch_to_restore
            # Create a net with empty input/val just to be able to predict without training
            self.create_net( np.empty( data['input_data_shape'] ), 
                             np.empty( data[ 'val_data_shape' ] ) )
            # Reload the error scores up to the epoch required
            self.train_MEE = self.train_MEE[:self.epoch]
            self.train_MSE = self.train_MSE[:self.epoch]
            self.val_MEE = self.val_MEE[:self.epoch]
            self.val_MSE = self.val_MSE[:self.epoch]


    def __getattr__(self,attr):
        """Get the atribute of MLP"""
        return [getattr(lay,attr) for lay in self.network]

    def train(self, input_data, labels, val_data, val_labels, epoch,
              eta=0.1, eta_params = None, lamb=0, norm_L=2, alpha=0, 
              nesterov=False, clean_net = False, save_rate = -1, batch_size=-1,
              filename = None, verbose=True, print_rate = 10, 
              RMSProp=False, beta=0.8, error_threshold = 0., patience = -1):
        """
        Parameters
        ----------
        input_data : list or numpy 2d array
            Array with the data in input to the MLP.
        labels : list or numpy 2d array
            Labels of the input_data.
        val_data : list or numpy 2d array
            Array with the validation data in input to the MLP.
        val_labels : list or numpy 2d array
            Labels of the val_data.
        epoch : int
            Number of epoch for training.
        eta : float or function obect
            The learning rate (default is 0.1), if this parameter is a 
            function than the parameter need to be passed in eta_params.
            The parameters need to be in the following order:
            eta(par_1, par_2, ..., par_n, attr_1 = None, ..., attr_n = None)
            - par_i are external parameters (constant value)
            - attr_i are the attribute of MLP class on which the function eta
              depends.
        eta_params: list or None
            If eta is a function this argument is the list of parameters of 
            that function. The parameters have the following structure:
            list = [*func_args, *func_kwargs]
            - funct_args contains the constant parameters of the function: 
              all the parameters that don't change during the training.
            - func_kwargs contains strings with the names of the attribute 
              on which the function eta depends. This attribute can change
              during the training so they will be updated at each training step.
        lamb : float
            The Tikhonov regularization factor (default is 0).
        norm_L : int
            The type of the norm for the Tikhonov regularization (default is 2,
            euclidean).
        alpha : int
            Momentum factor. Default is 0.
        nesterov = False
            If True use the accelerated (Nesterov) momentum. Default is False.
        clean_net : boolean
            If True restore the net, if False keep the pretrained net (if exist)
            Default is False.
        save_rate : int
            Rate of saving of the net during the training (save every save_rate 
            epoch), if -1 don't save the net. Default is -1.
        batch_size: int
            The size of the training batch. If -1 implement a full-batch, if 
            1 implement sgd. For intermediate value implement a mini-batch.
        filename : string or None
            If save_rate != -1 the net will be saved in the dir 'filename'.
            If None then save the net in the current dir. Default is None.
        verbose : bool
            If verbose = True print the training stats.
        print_rate : int
            If verbose = True then print the training stats every print_rate 
            epoch. Default is 10.
        RMSProp: boolean
            If RMSProp = True than optimize the training with RMSProp method.
            Default is False.
        beta: float
            Parameter of RMSProp. Default is 0.8
        error_threshold: float
            Threshold for the patience during the training.
            This is the difference of error in the last two step divided
            by the total difference of error during all the training, all 
            multiplied by the number of epochs. If this value is lower than
            error_threshold then reduce the counter of patience. 
            When the counter reach zero interrupt the training.
            Default is 0.
        patience : int
            If the threshold is not respected for patience time then interrupt
            the training.
        """
        # Check if all the input are numpy arrays
        input_data = np.array(input_data)
        labels = np.array(labels)
        val_data = np.array(val_data)
        val_labels = np.array(val_labels)
        self.RMSProp = RMSProp
        self.beta = beta

        self.lamb = lamb # Tikhonov regularization
        self.norm_L = norm_L # Tikhonov regularization norm
        self.alpha = alpha # Parameter for momentum
        self.nesterov = nesterov # If the user want the nesterov momentum

        if callable(eta): # if eta is a function
            if eta_params == None: # if the user pass no parameters
                raise TypeError('Try to set eta as a function without parameters')
            else:
                self.eta_keys = [k for k in eta_params 
                                 if isinstance(k, str)] # attribute names
                self.eta_params = [p for p in eta_params 
                                   if isinstance(p, (int, float))] # values
                self.eta_function = eta # eta is a function now
        else: # Eta is not a function
            self.eta = eta # then eta is just a number
            self.eta_params = None # Just for discriminate from prev. case

        # Reset the net if clean_net == True
        if clean_net:
            self.train_MEE = []
            self.train_MSE = []
            self.val_MEE = []
            self.val_MSE = []
            self.train_accuracy=[]
            self.val_accuracy=[]
            self.epoch = 0

        # If the net is just an empty list fill it with the layers
        if len(self.network) == 0:
            self.create_net(input_data, val_data)
        else:
            self.network[0].input = input_data
            self.network[0].val_data = val_data

        if self.preloaded: # If is a preloaded net fill the input
            self.network[0].input = input_data
            self.network[0].val_data = val_data
        if batch_size == -1:
            batch_size = input_data.shape[0]
        if save_rate != -1 and filename == None:
            # If the user want to save the net but he doesn't specify the dir
            filename = './' # save in the current dir
        # Start train the net
        total_time = 0
        real_start = time.time()
        calm = patience
        string_err = ""
        for i in range(epoch):

            start_loop = time.time()

            #### Set up eta (eventually as a function) ####
            if self.eta_params != None: # if None eta is just a number
                # Get the name of the attribute and his value in a dictionary
                # for each attribute in the eta_keys (the string passed by user)
                eta_dict = {name_attr:getattr(self, name_attr) for name_attr in self.eta_keys}
                # Call the eta function with that parameters
                self.eta = self.eta_function(*self.eta_params, **eta_dict)

            #### Train with the mini batch ####
            if batch_size!=input_data.shape[0]:
                shuffle(input_data, labels)
                for tr, lab in mini_batch(input_data,labels,batch_size):
                    self.network[0].input = tr
                    self.feedforward()
                    self.learning_step(lab)
                # Train dataset #
                self.network[0].input = input_data
                self.feedforward()
            else:
                self.feedforward()
                self.learning_step(labels)
            MEE, MSE = MEE_MSE(labels,self.network[-1].out)
            self.train_MEE.append(MEE)
            self.train_MSE.append(MSE)
            self.train_accuracy.append(1-np.sum(np.abs(np.heaviside(self.network[-1].out-0.5,0)-labels))/len(labels))
            # Validation dataset #
            self.feedforward_validation()
            MEE, MSE = MEE_MSE(val_labels, self.network[-1].out_val)
            self.val_MEE.append(MEE)
            self.val_MSE.append(MSE)
            self.val_accuracy.append(1-np.sum(np.abs(np.heaviside(self.network[-1].out_val-0.5,0)-val_labels))/len(val_labels))

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
            if verbose and self.epoch % print_rate == 0:
                print(f'[Epoch {self.epoch}]' + string_err + string_time + ' '*10, end = '\r', flush=True)
            if (i%save_rate==0) and (save_rate >= 0):
                self.save_network(filename)

            # Updating epoch
            self.epoch += 1
 
            if (self.epoch > 2) and (patience != -1):
                total_descend_train = self.train_MSE[0] - self.train_MSE[-1]
                total_descend_val = self.val_MSE[0]- self.val_MSE[-1]
                last_descend_train  = (self.train_MSE[-2] - self.train_MSE[-1])*self.epoch
                last_descend_val  = (self.val_MSE[-2] - self.val_MSE[-1])*self.epoch
                last_on_tot_train = last_descend_train/total_descend_train
                last_on_tot_val = last_descend_val/total_descend_val
                if (last_on_tot_train < error_threshold) or (last_on_tot_val < error_threshold):
                    calm = calm-1 # The calm is finishing...
                else: 
                    calm = min(patience, calm + 1)
            if calm == 0: break # Lost the patience: stop

        # Final print
        if verbose:
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

    def create_net(self, input_data, val_data):
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
                w = data[f'weights_epoch{self.epoch}']
                bias = data[f'bias_epoch{self.epoch}']
        for layer,num_unit in enumerate(self.structure):
            if layer==0: in_train, in_val = input_data, val_data
            else: 
                in_train = self.network[layer-1].out 
                in_val   = self.network[layer-1].out_val
            # Define the layer
            lay = Layer(num_unit, in_train, val_matrix=in_val,
                        starting_points = self.starting_points[layer],
                        func=self.func[layer], preload_w = w[layer],
                        preload_bias = bias[layer], from_backup = from_backup)
            self.network.append(lay) # add the layer

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
        num_data=labels.shape[0]
        for reverse_layer_number,layer in enumerate(self.network[::-1]):
            if self.nesterov:
                layer.dw_nest=self.alpha*layer.dW_1
            else:
                layer.dw_nest=0
            if reverse_layer_number==0:
                delta=((labels-layer.out)*layer.der_func(layer.net_nest))/num_data
            else:
                delta=(np.matmul(delta,weight_1)*layer.der_func(layer.net_nest))
            weight_1=layer.weight+layer.dw_nest

#xxxxxxxxxxx Comment here for  5x speed up xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
#            grad_W=np.sum([np.outer(i,j) for i,j in zip(delta,layer.input)], axis=0) #batch
#            grad_b=np.sum(delta,axis=0)
#xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

#%%%%%%%%%%% Decomment here for 5x speed up %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            grad_W, grad_b = self._jit_update_weight(delta, layer.input)
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            if self.RMSProp:
                dW,db=layer.RMSProp(grad_W,grad_b,self.eta,self.beta)
                dW-=self.lamb * (np.abs(layer.weight)**(self.norm_L-1))*np.sign(layer.weight)
            else:
                dW = (self.eta)*grad_W - self.lamb * (np.abs(layer.weight)**(self.norm_L-1))*np.sign(layer.weight)
                db = (self.eta)*grad_b
            layer.weight+=dW+self.alpha*layer.dW_1
            layer.bias  +=db+self.alpha*layer.db_1
            layer.dW_1 = dW
            layer.db_1 = db
#%%%% Decomment the block for a 5x speed up %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    @staticmethod
    @njit(cache = True, fastmath = True)
    def _jit_update_weight(delta, inputs):
        dW = np.outer(delta[0], inputs[0])
        db = delta[0]
        for i in range(1, len(delta)): # speed up this loop on data with numba!
            dW += np.outer(delta[i], inputs[i])
            db += delta[i]
        return dW, db

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
            net_dict = self.__dict__.copy() # Dict of the MLP object
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

            # Overwrite the parameters that change during training
            # overwrite the val, train curves
            save_dict['train_MEE'] = self.train_MEE
            save_dict['train_MSE'] = self.train_MSE
            save_dict['val_MEE'] = self.val_MEE
            save_dict['val_MSE'] = self.val_MSE
            #overwrite the number of epoch
            save_dict['epoch'] = self.epoch

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

def mini_batch(input_data,labels, batch_size):
    for i in range(0,len(input_data),batch_size):
        yield input_data[i:i+batch_size,:], labels[i:i+batch_size,:]

def shuffle(input_data, labels):
    seed=np.random.randint(2**32-1)
    rand_state = np.random.RandomState(seed)
    rand_state.shuffle(input_data)
    rand_state.seed(seed)
    rand_state.shuffle(labels)



class MLP_w(MLP):
    """
    MultiLayerPerceptron Class with different weights initialization.
    """
    def __init__(self, structure = [], func = None, starting_points=None,
                 filename = None, epoch_to_restore = -1):
        # Create all the attribute from MLP class
        super().__init__(structure = structure, func = func, 
                         starting_points=starting_points, 
                         filename = filename, epoch_to_restore = epoch_to_restore)

    def train(self, *args, n_candidate = 1, test_more_init = False, **kwargs):
        if test_more_init: # If the user want to test more random initial conditions
            # Initialize some MLP_w parameters
            self.candidate = None
            self.best_candidate = None
            self.best_error = np.infty
            self.candidate_error = None
            for i in range(n_candidate): # loop on the number of candidate
                print(f'Candidate {i}')
                # Initialize a candidate
                self.candidate = MLP(structure = self.structure, func = self.func, 
                                 starting_points=self.starting_points)
                self.candidate.train(*args, **kwargs) # Train a candidate
                # Save the last error
                self.candidate_error = self.candidate.val_MEE[-1]
                # If the last error is lower than the best error you find the new
                # best model!
                if self.candidate_error < self.best_error:
                    self.best_candidate = self.candidate # Save the best model
                    self.best_error = self.candidate_error # Save the best error
            dict_best = self.best_candidate.__dict__.copy() # Dict of the candidate object
            for key, val in dict_best.items(): # for each attribute of best_candidate
                setattr(self, key, val) # Overwrite the attribute of MLP_w with the 
                #                       # attribute of the best model
        else: # It the user don't want to test more weigths (maybe already done)
            super().train(*args, **kwargs) # Just train the model
