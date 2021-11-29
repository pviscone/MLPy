"""
Implementation of the Multi Layer Perceptron.
"""
import time
import os
import numpy as np
#from numba import njit # <---- Decomment for a 5x speed up
from utils.losses import MEE_MSE
from layer import Layer

class MLP:
    """
    Multi Layer Perceptron class.
    """
    def __init__(self, structure = [], func = None, starting_points = 0.1,
                 eta=0.1, lamb=0, norm_L=2, alpha=0, nesterov=False, directory_name = None):
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
        """
        self.directory_name = directory_name
        if self.directory_name == None:
            self.structure=structure #numero di unità per ogni layer
            # list of tuple with (function, parameter of funtcion)
            self.func=[f if isinstance(f, (tuple, list)) else (f, 1) for f in func]
            self.starting_points=starting_points #lista degli start_point
            self.eta = eta
            self.lamb = lamb
            self.norm_L = norm_L
            self.alpha = alpha
            self.nesterov = nesterov
        else:
            if self.directory_name[-1] != '/': self.directory_name += '/'
            with open(self.directory_name + 'net_info', 'r') as f:
                for line in f:
                    attr = line.split(' -> ')[0][2:]
                    raw_val = line.split(' -> ')[1][:-1]
                    if attr == 'structure':
                        struct = raw_val[1:-1].split(', ')
                        val = [int(num) for num in struct]
                    elif attr == 'func':
                        val = []
                        funcs = raw_val[1:-2].split('), ')
                        for fs in funcs:
                            f_name = fs.split(', ')[0][2:-1]
                            f_param = int(fs.split(', ')[1])
                            tuple_f = (f_name, f_param)
                            val.append(tuple_f)
                    elif attr == 'starting_points':
                        struct = raw_val[1:-1].split(', ')
                        val = [float(num) for num in struct]
                    elif attr == 'nesterov':
                        if line.split(' ->')[1][:-1] == 'False':
                            val = False
                        elif line.split(' ->')[1][:-1] == 'True':
                            val = True
                        else: val = False
                    else:
                        val = float(line.split(' ->')[1])
                    setattr(self, attr, val)

        self.network=[]
        self.train_MEE = []
        self.train_MSE = []
        self.val_MEE = []
        self.val_MSE = []
        self.epoch = 0


    def __getattr__(self,attr):
        """Get the atribute of MLP"""
        return [getattr(lay,attr) for lay in self.network]

    def train(self, input_data, labels, val_data, val_labels, epoch,
                clean_net = False):
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
            self.create_net(input_data,val_data)

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
            string_val_err = f'  [val MEE = {self.val_MEE[self.epoch]:.4f}]'
            string_err = f'  [train MEE = {self.train_MEE[self.epoch]:.4f}]'
            string_err += string_val_err

            # Printing remaining time
            elapsed_for_this_loop = time.time()-start_loop
            total_time += elapsed_for_this_loop
            mean_for_loop = total_time/(i+1)
            remain_time = mean_for_loop*(epoch-i)
            string_time = f'  [wait {remain_time:.1f} s]'
            print(f'[Epoch {self.epoch}]' + string_err + string_time + ' '*10, end = '\r', flush = True)

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

    def create_net(self, input_data, val_data):
        """
        Feed the input of the net and propagate it

        Parameters
        ----------
        input_data : list or numpy 2d array
            Array with the data in input to the MLP.
        """
        if self.directory_name == None:
            for layer,num_unit in enumerate(self.structure):
                if layer==0: #If empty, initializing the neural network
                    self.network.append(Layer(num_unit,input_data,
                                        val_matrix=val_data,
                                        func=self.func[layer],
                                        starting_points=self.starting_points[layer]))
                else:
                    self.network.append(Layer(num_unit,self.network[layer-1].out, 
                                        val_matrix=self.network[layer-1].out_val,
                                        func=self.func[layer],
                                        starting_points=self.starting_points[layer]))
        else:
            for layer,num_unit in enumerate(self.structure):
                bias_w = np.loadtxt(directory_name + f'layer{layer}.txt')
                bias = bias_w[:, 0]
                w = bias_w[:, 1:]
                if layer==0: #If empty, initializing the neural network
                    self.network.append(Layer(num_unit,input_data,
                                        val_matrix=val_data,
                                        func=self.func[layer],
                                        preload_w = w,
                                        preload_bias = bias))
                else:
                    self.network.append(Layer(num_unit,self.network[layer-1].out, 
                                        val_matrix=self.network[layer-1].out_val,
                                        func=self.func[layer],
                                        preload_w = w,
                                        preload_bias = bias))


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


    def save_network(self, directory_name):
        net_dict = self.__dict__
        skip_list = ['network', 'directory_name', 'train_MEE', 'train_MSE', 
                     'val_MEE', 'val_MSE', 'epoch']
        # Write general info of the net
        if not os.path.exists(directory_name):
            os.makedirs(directory_name)
        with open(directory_name + 'net_info', 'w') as f:
            for net_attr in net_dict.keys():
                if net_attr not in skip_list:
                    f.write(f'# {net_attr} -> {net_dict[net_attr]}\n')
        for i, layer in enumerate(self.network):
            np.savetxt(directory_name + f'layer{i}.txt', np.c_[layer.bias, layer.weight])
