import numpy as np
import os
import itertools # Include in standard python library
import time 
from . preprocessing import split
from joblib import Parallel, delayed


def grid_search(model, dict_models, dict_params, input_data, labels, error, 
                verbose = True, n_jobs = 1, **kwargs):
    """
    Function that implement a grid search algorithm for the model selection.

    Parameters
    ----------
    Model : Class Object
        Model for the task.
        The model need to have a function called "train" witch take some of the
        parameters in dict_params.
    dict_models : dictionary
        Dictionary containing the name of the models as keys and other 
        dictionaries as values. This dictionaries contain the parameters
        of the model to test.
    dict_params: dictionary
        Parameter of Model to test in the search, each key of the dictionary 
        is a parameter of model, the value corresponding to each key is a list
        of parameters to test for that key.
        The keys of the dictionary need to be written in the following way:
        - "model_params"
        - "train_params"
        The first kind of key denote a parameter of the model, the second 
        denote a parameter of the train function.
        The grid is builded with every possibile combination of that lists.
    input_data : list or numpy 2d array
        Array with the data in input to the MLP.
    labels : list or numpy 2d array
        Labels of the input_data.
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
    def train_one_combination(k, params):
        ######## Create the dictionary with parameters of candidate #######
        dict_candidate = {} # Fill with the info for each candidate model
        # Fill the train parameters (like {'p1':p1_val, 'p2':p2_val, ...} )
        dict_candidate['train'] = {k:p for k, p in zip(dict_params.keys(), params)}
        dict_candidate['model'] = mod_params # set the model parameters
        dict_candidate['model_name'] = mod_name # and the model name 
        # APPEND
        
        ###### Create and train the candidate #############################
        sum_err = 0
        if kwargs['kind'] == 'hold_out': k = 1
        elif kwargs['kind'] == 'k_fold': k = kwargs['k']
        else: raise Exception('Invalid split method')
        for _ in range(k):
            data_k, val_k, labels_k, val_labels_k = split(input_data, labels, **kwargs)
            candidate = model(**dict_candidate['model'])

            # Train the candidate with the parameters in dict_params['train']
            candidate.train(data_k, labels_k, val_k, val_labels_k, 
                    **dict_candidate['train'])
            # predict on validation data
            pred_val = candidate.predict(val_k)

            err_k =  error(val_labels_k, pred_val)
            sum_err += err_k
        # add the error to the list of error to position i
        # ATTENTION!!
#        print(f'in position {i} I put {sum_err/k}')
        return sum_err/k, dict_candidate
    if n_jobs == -1:
        n_jobs = os.cpu_count()

    # Extract the values of the dictionary
    list_params_values = list(dict_params.values())
    # Create a list with all the permutations of the values
    list_permutation = list(itertools.product(*list_params_values))
    list_candidates = [] # Initialize an empty list for save all the candidates
    list_results = [] # List for final sorting of candidates based on error
    i = 0 # incremental variable for double loop
    n_candidates = len(list_permutation)*len(dict_models) # num candidates
    err_val_array = []# np.empty(n_candidates) # Empty array for the errors
    n_models = len(dict_models)
    j = 0
    start = time.time() # Time counter
    tot_time = 0
    print(f'start training {n_candidates} possible combinations')
    for mod_name, mod_params in dict_models.items(): # Apply the combination to
        #                                            # each model
        inner_time = time.time()
        out = Parallel(n_jobs=n_jobs)(delayed(train_one_combination)(k, params) 
                                      for k, params in enumerate(list_permutation))
#        print(out)
        inner_list_E = [xx[0] for xx in out] # save the parallel error in a list
        inner_list_cand = [xx[1] for xx in out] # save the parallel candidate in a list
        # append the candidate to the final list
        [list_candidates.append(c) for c in inner_list_cand] # Add to list of candidates

        for e in inner_list_E: # Empty array for the errors
            err_val_array.append(e)

        elapsed_inner = time.time()-inner_time # time for single loop
        tot_time += elapsed_inner # total time up to j
        mean_tot_time = tot_time/(j+1)
        n_remaining = n_models - (j + 1)
        if verbose:
            print(f'[trained={(j+1) * len(list_permutation)}] [elaps t={tot_time:.1f} s] [remain t ={n_remaining * mean_tot_time:.1f} s]', end = '\r')
        j += 1

    err_val_array = np.array(err_val_array) # for np.sort...
    if verbose: print(f'\nTime for the grid search: {time.time()-start} s')

    # Sort the errors and return the index of the sorting 
    idx_sorted_val = np.argsort(err_val_array)

    for i in range(len(list_candidates)):
        # Extract the index using the sorted array of index
        real_idx = idx_sorted_val[i]
        # Add to the candidate the relative error
        list_candidates[real_idx]['Error'] = err_val_array[real_idx]
        # Add to the array of results the candidate in the chart order
        list_results.append(list_candidates[real_idx])
    return list_results
