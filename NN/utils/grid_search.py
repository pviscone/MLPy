import numpy as np
import itertools # Include in standard python library
import time 

def grid_search(model, dict_params, verbose = 0):
    """
    Function that implement a grid search algorithm for the model selection.

    Parameters
    ----------
    Model : Class Object
        Model for the task
    dict_params: dictionary
        Parameter of Model to test in the search, each key of the dictionary 
        is a parameter of model, the value correspondind to each key is a list
        of parameters to test for that key.
        The grid is builded with every possibile combination of that lists.
    verbose: int
        Parameter for having some print during the grid search... 
        This function requires a lot of time to run, better be patience and 
        put verbose 1.
    """
    dict_results = {}
    # Extract the values of the dictionary
    list_params_values = list(dict_params.values())
    # Exract the keys
    list_params_keys = list(dict_params.keys())
    # Create a list with all the permutations of the values
    list_permutation = list(itertools.product(*list_params_values))
    list_candidates = [] # Initialize an empty list for the final result
    for params in list_permutation: # For each combination of parameter
        # Create a dictionary that recreate the structure of the input 
        # to the model: {'p1':p1_val, 'p2':p2_val, ...}
        dict_candidate = {k:p for k, p in zip(list_params_keys, params)}
        # Append that dictionary to the final list
        list_candidates.append(dict_candidate)

    for i, dict_params in enumerate(list_candidates):
        print(f'{i} -> Parameters in dictionary are:', dict_params)
        print('       The function get:', model(**dict_params))
        # Define a model
#        candidate_model = model()# oh shit
        # train the model
#        candidate_model.train()# oh shit again
        # Some parameters are in train and other in the model definition.
        # Why dont' we put all in the model definition? 
        # - it is not good to change the parameters manually on the middle of
        #   the training (separate speech for epoch that need to stay in train)
        # - Manually changing parameters during the training is dangerous for
        #   reproducibility: we need to mark every single modification...
        # - Manually changing is not automatic: better an adaptive learning
        #   step that change after some epochs...
        # - This function became so easy if we put all in main:
        #   Just "candidate_model = (**dict_params)" 
        #   and then "candidate_model.train()"
    return 
