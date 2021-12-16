import numpy as np
import os
import itertools # Include in standard python library
import time 

def grid_search(model, dict_models, dict_params, input_data, labels, val_data, val_labels, error, verbose = False):
    """
    Function that implement a grid search algorithm for the model selection.

    Parameters
    ----------
    Model : Class Object
        Model for the task.
        The model need to have a function called "train" wich take some of the
        parameters in dict_params.
    dict_models : dictionary
        Dictionary containing the name of the models as keys and other 
        dictionaries as values. This dictionaries contanin the parameters
        of the model to test.
    dict_params: dictionary
        Parameter of Model to test in the search, each key of the dictionary 
        is a parameter of model, the value correspondind to each key is a list
        of parameters to test for that key.
        The key of the dictionary need to be written in the following way:
        - "model_params"
        - "train_params"
        The first kind of key denote a parameter of the model, the second 
        denote a parameter of the train function.
        The grid is builded with every possibile combination of that lists.
    input_data : list or numpy 2d array
        Array with the data in input to the MLP.
    labels : list or numpy 2d array
        Labels of the input_data.
    val_data : list or numpy 2d array
        Array with the validation data in input to the MLP.
    val_labels : list or numpy 2d array
        Labels of the val_data.
    error: function object
        Error function for define the chart of models, this function take 2 
        parameters: the real labels and the output (predicted) labels.
    verbose: boolean
        Parameter for having some print during the grid search... 
        This function requires a lot of time to run, better be patience and 
        put verbose True.

    Returns
    -------
    List of dictionary
        List of dictionary corresponding to the chart of models with respect
        to the error given in input. Each dictionary contains the items:
        - 'train': the train parameters of the model.
        - 'model': the parameters of the model.
        - 'model_name': the name of the model given in dict_models.
    """
    list_results = []
    # Extract the values of the dictionary
    list_params_values = list(dict_params.values())
    # Exract the keys
    list_params_keys = list(dict_params.keys())
    # Create a list with all the permutations of the values
    list_permutation = list(itertools.product(*list_params_values))
    list_candidates = [] # Initialize an empty list for the final result
    # This loop is for flattening all the candidate in an unique list 
    # (to perform an easy sort on error then).
    for mod_name, mod_params in dict_models.items():
        for params in list_permutation: # For each combination of parameter
            # Create a dictionary that recreate the structure of the input 
            # to the model: {'p1':p1_val, 'p2':p2_val, ...}
            dict_candidate = {}
            # Fill the train parameters
            dict_candidate['train'] = {k:p for k, p in zip(list_params_keys, params)}
            dict_candidate['model'] = mod_params # set the model parameters
            dict_candidate['model_name'] = mod_name # and the model name 
            # Append that dictionary to the final list
            list_candidates.append(dict_candidate)
       
    n = len(list_candidates) # Number of candidates
    err_val_array = np.empty(n) # Empty array for the errors
    start = time.time() # Time counter
    for i, dict_params in enumerate(list_candidates): # for each candidate
        mod_params = dict_params['model'] # Extract model params'
        mod_name = dict_params['model_name'] # Extract model name
        candidate = model(**mod_params) # Create the candidate
        if verbose: # Print only if verbose
            print(f"{i} on {n}: Model {mod_name} --> Train parameters: {dict_params['train']}")
        # Train the candidate with the parameters in dict_params['train']
        candidate.train(input_data, labels, val_data, val_labels, **dict_params['train'])
        # predict on validation data
        pred_val = candidate.predict(val_data)
        # add the error to the list of error to position i
        err_val_array[i] = error(val_labels, pred_val)
        print('\n')
    print(f'Time for the grid search: {time.time()-start} s')

    # Sort the errors and return the index of the sorting 
    order_validation = np.argsort(err_val_array)

    for i in range(n):
        # Extract the index using the sorted array of index
        real_idx = order_validation[i]
        # Add to the candidate the relative error
        list_candidates[real_idx]['Error'] = err_val_array[real_idx]
        # Add to the array of results the candidate in the chart order
        list_results.append(list_candidates[real_idx])
    return list_results
