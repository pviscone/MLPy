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
        labels = norm(labels)
        val_labels = norm(val_labels)
        train_pred = norm(train_pred)
        val_pred = norm(val_pred)

    '''
    if sortidx != None:
        sorted_index = np.argsort(train_pred[:, sortidx])
        train_pred = train_pred[sorted_index, :]
        labels = labels[sorted_index,:]

        sorted_index = np.argsort(val_pred[:, sortidx])
        val_pred = val_pred[sorted_index, :]
        val_labels = val_labels[sorted_index,:]'''

    if sortidx != None:
        val_labels_x = np.sort(val_labels[:,1])
        argsort=np.argsort(val_labels[:,1])
        val_labels_y = val_labels[argsort,0]
        val_pred_x=val_pred[argsort,1]
        val_pred_y=val_pred[argsort,0]
        val_res_x=val_pred_x-val_labels_x
        val_res_y=val_pred_y-val_labels_y

        train_labels_x = np.sort(labels[:,1])
        argsort=np.argsort(labels[:,1])
        train_labels_y = labels[argsort,0]
        train_pred_x=train_pred[argsort,1]
        train_pred_y=train_pred[argsort,0]
        train_res_x=train_pred_x-train_labels_x
        train_res_y=train_pred_y-train_labels_y

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
    plt.plot(train_labels_x, train_res_x, ".",label="Residual output 1")
    plt.plot(train_labels_x, train_res_y, ".",label="Residual output 0")
    plt.legend()

    fig.add_subplot(133)
    plt.title('Residual for validation data')
    plt.plot(val_labels_x, val_res_x, ".",label="Residual output 1")
    plt.plot(val_labels_x, val_res_y, ".",label="Residual output 0")
    plt.legend()
    plt.tight_layout()
    plt.show()
    print('final train error:', MEE(labels, train_pred))
    print('final val error:', MEE(val_labels, val_pred))


def output_correlations(model, data, labels, fit_func = None, func_args = None, mean_fit = True, plot_arrow_worse = None):
    pred=model.predict(data)
    x = pred[:,1]
    y = pred[:,0]
    lab_x = labels[:,1]
    lab_y = labels[:,0]
    if fit_func != None:
        fit_out = fit_func(x, *func_args)
        mean_out = (fit_out + y)/2
        if mean_fit: y = mean_out
        else: y = fit_out

    plt.figure(figsize=(14,4))
    plt.subplot(121)
    plt.title('original vs predicted')
    plt.scatter(lab_x, lab_y, s = 4, label = 'original')
    plt.scatter(x, y, s = 4, label = 'predicted')
    plt.legend()
    
    plt.subplot(122)
    plt.title('Heatmap based on MEE error')
    c = np.sqrt(np.sum((labels - pred)**2, axis = 1))
    plt.scatter(lab_x, lab_y, s = 4, alpha = 0.3, label = 'original')
    plt.scatter(x, y, s = 4, cmap = 'Reds', c = c)
    plt.colorbar()
    plt.legend()
    plt.show()

    if plot_arrow_worse != None:
        plt.figure(figsize=(14,6))
        n = plot_arrow_worse
        plt.title(f'Where should go the worst {n} points')
        labels_pred = np.column_stack((y, x))
        s = np.argsort(c) 
        worse = s[-n:]
        for pr, re in zip(labels_pred[worse, :],labels[worse,:]):
            d = np.sqrt((pr[0] - re[0])**2 + (pr[1]-re[1])**2)
            plt.arrow(pr[1], pr[0], re[1]-pr[1], re[0]-pr[0], lw = 0.5, alpha = 0.5,
                    head_width = d*0.1, color = 'red', length_includes_head = True, fill = False)
        plt.scatter(lab_x[worse], lab_y[worse], s = 10, alpha = 1, c = 'blue')
        plt.scatter(lab_x, lab_y, s = 1, alpha = 0.1, c = 'blue')
        plt.scatter(x[worse], y[worse], s = 10, alpha = 1, c = 'orange')
        plt.scatter(x, y, s = 1, alpha = 0.1, c = 'orange')
        plt.show()
        return worse
