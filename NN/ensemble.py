import numpy as np

class bagging_ensemble:
    """
    Class that averages the results of more classifier in order to 
    reduce the variance of a final model.
    The bootstrap algorithm of resampling is applied to train a certain 
    number of same model. More kind of model can be combine also applying 
    more time the boostrap algorithm.
    Reference:
    - https://www.sciencedirect.com/science/article/pii/S000437020200190X
    """
    def __init__(self, repetition):
        """
        __init__ of ensemble class.
        
        Parameters
        ----------
        repetition : int
            Number of repetition for bootstrap algorithm.
        """
        self.repetition = repetition
        self.list_classifier = []
        
    def train(self, data, labels, val_data, val_labels, classifier, dict_classifier, 
              dict_train, bootstrap = True):
        """
        Train function of the ensemble.

        Parameters
        ----------
        data: numpy nd array 
            The dataset to train.
        labels: numpy nd array
            The output labels of the data.
        val_data: numpy nd array
            The validation dataset.
        val_labels: numpy nd array
            The labels of the validation dataset.
        classifier: class object
            Class that contains the methods:
                - train(input_data, labels, val_data, val_labels,**kwargs)
                - predict(data)
            and the property val_MEE, train_MEE with the relative error during
            the training.
            An example of this model is the MLP.MLP model.
        dict_classifier: dictionary
            Dictionary that contain the parameters of the __init__ function of 
            the classifier.
        dict_train : dictionary
            Dictionary that contain the parameters of the train function of 
            the classifier.
        bootstrap: boolean
            Boolean value for the bootstrap resampling: if True then a bagging
            ensemble is performed, otherwise the class is a simple model 
            averaging ensemble.
        """
        # Number of classifier already present in the ensemble
        prev = len(self.list_classifier)
        all_data = np.column_stack((data, labels)) # Merge all data together for bootstrap
        if bootstrap: 
            # Generate a generator of resampling
            bootstrap_gen = self.bootstrap(all_data, self.repetition)
            for i, all_d in enumerate(bootstrap_gen): # for each sampling in the generator
                print(f'{i + prev}/{self.repetition + prev}')
                # train and append the classifier to the list of classifier
                c = classifier(**dict_classifier)
                training = all_d[:,:-labels.shape[1]]
                train_labels = all_d[:,-labels.shape[1]:]
                c.train(training, train_labels, val_data, val_labels, **dict_train)
                self.list_classifier.append(c)
        else:
            # just train and append self.repetition time on the same dataset
            for i in range(self.repetition):
                print(f'{i + prev}/{self.repetition + prev}')
                c = classifier(**dict_classifier)
                c.train(data, labels, val_data, val_labels, **dict_train)
                self.list_classifier.append(c)

        self.corr_matrix(data, labels) # evaluate the correlation matrix

    def predict(self, data, func=None, param=None):
        """
        Predict function.
        
        Parameters
        ----------
        data : numpy nd array
            Array of which the function predict the output.
        func : function object
            Function for eventual fit of the second output 
            given the first one (correlation in the ML CUP)
            The free variable is x, the function parameters 
            are in the form: func(x, *params)
        param : list
            List of parameters of the function func (excluded x)
        """
        out = 0
        # evaluate the mean of the predictions
        for c in self.list_classifier:
            out += c.predict(data)
        out=out/len(self.list_classifier)
        # if func is not None
        if func != None:
            # apply the average of the function out and the second output
            out[:,0]+=func(out[:,1],*param)
            out[:,0]/=2
        return out
    
    def bootstrap(self, data, repetition = 20, sample_size = None):
        """
        Bootstrap resampling generator.
        """
        # If sample_size is None than the size of the resample is the same of data
        if sample_size == None: sample_size = len(data)
        # number of subsample is the fixed by the parameter repetition
        n_subsample = np.floor(len(data)/repetition).astype(int)
        rng = np.random.default_rng() # Initialize a random generator
        for k in range(repetition):
            # Fraction of data subsampled
            subsample = data[n_subsample * k:min(n_subsample * (k+1), sample_size) ]
            # expand random the subsample from uniform distribution
            yield rng.choice(subsample, sample_size, axis = 0)
            
    @property
    def train_MEE(self):
        """
        Mean of the train_MEE curve, not so significant parameter.
        """
        min_len = np.min([len(c.train_MEE) for c in self.list_classifier])
        
        train_MEE = np.array(self.list_classifier[0].train_MEE[:min_len])
        for c in self.list_classifier[1:]:
            train_MEE += np.array(c.train_MEE[:min_len])
        return train_MEE/len(self.list_classifier)
    @property
    def val_MEE(self):
        """
        Mean of the val_MEE curve, not so significant parameter.
        """
        min_len = np.min([len(c.val_MEE) for c in self.list_classifier])
        val_MEE = np.array(self.list_classifier[0].val_MEE[:min_len])
        for c in self.list_classifier[1:]:
            val_MEE += np.array(c.val_MEE[:min_len])
        return val_MEE/len(self.list_classifier)

    def corr_matrix(self, data, labels):
        """
        Correlation matrix between classifier.
        """
        n = len(self.list_classifier)
        C = np.empty((labels.shape[1], n,n))
        for i, c_i in enumerate(self.list_classifier):
            for j, c_j in enumerate(self.list_classifier[:i+1]):
                res_i = c_i.predict(data) - labels
                res_j = c_j.predict(data) - labels
                for k in range(labels.shape[1]):
                    C[k, i, j] = np.sum(res_i[:,k] * res_j[:,k])
                    C[k, j, i] = C[k, i, j]
        self.C = C
        return np.squeeze(C)

    @property
    def prune_ensemble(self):
        """
        Function that prune redundant (correlated) classifier.
        """
        n = len(self.list_classifier)
        fact1 = (2 * n - 1 ) * np.sum(self.C, axis = (1,2) ) 
        new_list_classifier = []
        new_C = self.C
        j = 0
        for i in range(n):
            mask = np.ones(new_C.shape[1]).astype(bool)
            mask[j] = False
            fact2 = 2 * n*n * np.sum(new_C[:,mask,j], axis = 1) + n*n * self.C[:, j, j ]
            if (fact1 > fact2).any():
                new_list_classifier.append(self.list_classifier[i])
                j = j + 1
            else:
                new_C = new_C[:, mask, :][:,:,mask]
            if j == n-1 and len(new_list_classifier) == 0:
                raise Exception('All the classifier are strongly correlated')

        if len(new_list_classifier) == 0:
            raise Exception('All the classifier are strongly correlated')
        else: 
            print(f'Pruned {n - len(new_list_classifier)}/{n} classifier')
            self.pruned_list_classifier = new_list_classifier
            self.C = new_C
