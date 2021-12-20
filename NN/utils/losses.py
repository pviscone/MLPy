"""
Losses function for model training and validation.
"""

import numpy as np

error = lambda label, out : np.sum( ( label - out )**2 )

def MEE_MSE(labels, out):
    """
    Mean Euclidean Error and Mean Squared Error functions.

    Parameters
    ----------
    labels : numpy 1d array or numpy 2d array
        Array containing the labels of the dataset.
    out : numpy 1d array or numpy 2d array
        Array containing the output labels predicted by the NN.

    Returns
    -------
    (float, float)
        (MEE, MSE) of the model.
    """
    inv_n_data = 1/len(labels) # "Multiply is cheaper than divide" (Mannella docet.)
    err_each_data = np.sum((labels - out)**2, axis = 1)
    MEE = np.sum(np.sqrt(err_each_data))*inv_n_data
    MSE = np.sum(err_each_data)*inv_n_data
    return MEE, MSE

def MEE(labels, out):
    """
    Mean Euclidean Error.

    Parameters
    ----------
    labels : numpy 1d array or numpy 2d array
        Array containing the labels of the dataset.
    out : numpy 1d array or numpy 2d array
        Array containing the output labels predicted by the NN.

    Returns
    -------
    (float, float)
        (MEE, MSE) of the model.
    """
    inv_n_data = 1/len(labels) # "Multiply is cheaper than divide" (Mannella docet.)
    err_each_data = np.sum((labels - out)**2, axis = 1)
    MEE = np.sum(np.sqrt(err_each_data))*inv_n_data
    return MEE

def MSE(labels, out):
    """
    Mean Euclidean Error and Mean Squared Error functions.

    Parameters
    ----------
    labels : numpy 1d array or numpy 2d array
        Array containing the labels of the dataset.
    out : numpy 1d array or numpy 2d array
        Array containing the output labels predicted by the NN.

    Returns
    -------
    (float, float)
        (MEE, MSE) of the model.
    """
    inv_n_data = 1/len(labels) # "Multiply is cheaper than divide" (Mannella docet.)
    err_each_data = np.sum((labels - out)**2, axis = 1)
    MSE = np.sum(err_each_data)*inv_n_data
    return MSE

