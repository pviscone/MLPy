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
    (numpy 2d array, numpy 2d array)
        (train data matrix, validation data matrix)
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
        pass
