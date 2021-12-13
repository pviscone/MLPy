import numpy as np
import itertools
import time

def grid_search(model, dict_params, verbose = 0):
    dict_results = {}
    # Extract the values of the dictionary
    list_params_values = list(dict_params.values())
    # Exract the keys
    list_params_keys = list(dict_params.keys())
    # Create a list with all the permutations
    list_permutation = list(itertools.product(*list_params_values))
    list_candidates = []
    for params in list_permutation:
        dict_candidate = {}
        for k, p in zip(list_params_keys, params):
            dict_candidate[k] = p
        list_candidates.append(dict_candidate)
        print(params)
#    for dict_params in list_candidates:
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
        #   Just "candidate_model = ()"
    return 
