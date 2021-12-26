import numpy as np
import matplotlib.pyplot as plt
from .losses import MEE

def plot_results(network, input_data, val_data, 
                 labels, val_labels, norm = None, func = None, func_args = None, 
                 fit_idx = 0, new_out = False, mean_fit = False, 
                 sortidx = None):
    """
    Function to show the results of a regressor (network).

    Parameters
    ----------
    network : class object
        The regressor, this must have the method predict(data).
    input_data : numpy 2d array
        The dataset in input to the regressor.
    val_data : numpy 2d array
        The validation dataset.
    labels : numpy 2d array
        Output labels of the input_data.
    val_labels : numpy 2d array
        Output labels of the val_data.
    func : function object
        Function to predict the second output given the first.
        Needs to be in the form func(x, func_args).
    func_args : tuple
        Argument of the function func (excluded the data x).
    fit_idx : int
        Index in wich put the new output from the fit func.
    new_out : boolean
        If the output of the function func needs to be appended
        to the output of the regressor "network".
    mean_fit : boolean 
        If the output of the function func needs to be averaged with 
        the output of the network ad index "fit_idx"
    sortidx : int
        Index for sorting the ouputs residuals (see if in some out region
        the regressor predict worse).
    """
    train_pred = network.predict(input_data)
    val_pred = network.predict(val_data)
    if func!=None:
        fitted_out_train = func(train_pred[:,-1], *func_args)
        fitted_out_val = func(val_pred[:,-1], *func_args)
        if new_out:
            train_pred = np.insert(train_pred, fit_idx, fitted_out_train, axis = 1)
            val_pred = np.insert(val_pred, fit_idx, fitted_out_val, axis = 1)
        else:
            if mean_fit:
                train_pred[:, fit_idx] = (train_pred[:,fit_idx] + fitted_out_train)/2
            else: train_pred[:, fit_idx] = fitted_out_train
            
            if mean_fit:
                val_pred[:, fit_idx] = (val_pred[:,fit_idx] + fitted_out_val)/2
            else:
                val_pred[:, fit_idx] = fitted_out_val
    if norm!= None:
        train_pred = norm(train_pred)
        val_pred = norm(val_pred)

    if sortidx != None:
        sorted_index = np.argsort(train_pred[:, sortidx])
        train_pred = train_pred[sorted_index, :]
        labels = labels[sorted_index,:]

        sorted_index = np.argsort(val_pred[:, sortidx])
        val_pred = val_pred[sorted_index, :]
        val_labels = val_labels[sorted_index,:]

    x = np.arange(len(network.train_MEE))

    fig = plt.figure(figsize=(13,4))

    fig.add_subplot(131)
    plt.plot(x,network.train_MEE)
    plt.plot(x,network.val_MEE,label="test")
    plt.title("Learning curve")
    plt.xlabel("Epochs")
    plt.ylabel("Squared error")
    plt.yscale("log")
    plt.legend()

    fig.add_subplot(132)
    plt.title('Residual for training data')
    for i in range(labels.shape[1]):
        plt.plot(np.arange(len(labels)),labels[:,i]-train_pred[:,i],".",label=f"residual{i}")
    plt.legend()

    fig.add_subplot(133)
    plt.title('Residual for validation data')
    for i in range(val_labels.shape[1]):
        plt.plot(np.arange(len(val_labels)),val_labels[:,i]-val_pred[:,i],".",label=f"residual{i}")
    plt.legend()
    plt.tight_layout()
    plt.show()
    print('final train error:', MEE(labels, train_pred))
    print('final val error:', MEE(val_labels, val_pred))
