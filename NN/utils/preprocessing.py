"""
Function for the dataset splitting.
"""

import numpy as np

def split(input_matrix, frac_training=0.8, shuffle=False, kind="hold_out", k=4):
    """
    Splitting the data in three different set, one for training, one for
    validation, one for test. Each pattern of each set is selected randomly from
	the initial input_matrix.

    Parameters
    ----------
    input_matrix : numpy 2d array
        Input data matrix (containing also the labels) without the test set.
    frac_training : float
        Fraction of data to use in training, default is 0.8 .
    shuffle : bool
        If True shuffle the input matrix, default is False.
    kind : string
        Kind of splitting, default is 'hold_out'.
    k : int
        Number of fold for k-fold splitting.

    Returns
    -------
    (numpy 2d array, numpy 2d array) if kind = 'hold-out'
    (numpy 2d array,  list of tuple) if kind = 'k-fold'
        If hold out return (train data matrix, validation data matrix).
        If k-fold return (data matrix, list of tuple containing 2 indexs).
        The index in the list represent the (idx1 = start, idx2 = stop) index
        to split for each fold. In order to not waste memory you can separate
        with this 2 index each train-val dataset before training the model.
        Note:
            to extract validation set just use the slicing: data[idx1, idx2]
            to extract train set just use
            numpy.delete(data, slice(idx1,idx2), axis = 0)

    """
    copy_data = np.copy(input_matrix) # Copy the dataset to not change original
                                      # input_matrix
    if shuffle:
        np.random.shuffle(copy_data)
    if kind=="hold_out":
        idx=round(len(copy_data)*frac_training)
        training_set=copy_data[0:idx,:]
        validation_set=copy_data[idx:,:]
        return training_set, validation_set
    elif kind=="k-fold":
        n_data = len(copy_data)
        fold_size = np.floor(len(copy_data)/k)
        idxs = [(int(i*fold_size),
                 int((i+1)*fold_size))
                for i in range(k-1)]
        np.append(idxs, ((k-1)*fold_size, n_data))

        return copy_data, idxs

class Normalize:
    def normalize(self,input_matrix):
        self.mean=np.mean(input_matrix,axis=0)
        self.std=np.std(input_matrix,axis=0)
        return (input_matrix-self.mean)/self.std
    def denormalize(self,input_matrix):
        return input_matrix*self.std+self.mean
    def minmax(self,input_matrix):
        self.max=np.max(input_matrix,axis=0)
        self.min=np.min(input_matrix,axis=0)
        return (input_matrix-self.min)/(self.max-self.min)
    def deminmax(self,input_matrix):
        return input_matrix*(self.max-self.min)+self.min
