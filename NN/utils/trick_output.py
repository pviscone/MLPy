import numpy as np 
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def sinusoidal(x, a, w, phi, shift, m, q, stop1 = None, stop2 = None, stop = True):
    l = linear1(x, m, None, q0 = q)
    g = a*l*np.sin(w*x + phi) + shift # the parabola
    if stop: # Manual double heaviside
        g[x<stop1] = 0
        g[x>stop2] = 0
    return g

def linear1(x, m, func, *func_args, stop_left = None, stop_right = None,
           cut_left = False, cut_right = False, q0 = None):
    # Compute q in a way to create a continuos function
    if cut_left: 
        q = func(stop_left, *func_args, stop = False) - m * stop_left
    elif cut_right: 
        q = func(stop_right, *func_args, stop = False) - m * stop_right
    else: q = q0
    y = m*x + q # the linear function
    if cut_left: y[x<stop_left] = 0 # Manual left heaviside
    if cut_right: y[x>stop_right] = 0 # Manual right heaviside
    return y

def lin_sin_lin(x, a, w, phi, shift, m, q, m1, stop1, m2, stop2):
    """
    Fit function create by hand:
    - on the left a linear function with angolar coefficient m1
    - on the middle a sinusoid
    - on the right another linear function with m2
    stop1 and stop2 are the two points where you switch from 
    linear1->parabola and parabola->linear 2 (with continuity).
    NOTE: the intercept q of the linear function are fixed for the 
          continuity.
    """
    func = sinusoidal
    func_args = (a, w, phi, shift, m, q,)
    return linear1(x, m1, func, *func_args, stop_right = stop1, cut_right = True) +\
           func(x, *func_args, stop1, stop2) +\
           linear1(x, m2, func, *func_args, stop_left = stop2, cut_left = True)

def trick_params(norm_name = None):
    """
    Function that return the parameters for the convergence of the fit 
    discover in ML CUP.
    
    Parameters
    ----------
    norm_name : string
        The name of the normalizer if the dataset is normalized.
        Avaible norm are:
            - None : no normalization
            - 'std': StandarScaler normalization.
            - 'minmax': MinMax normalization.

    Returns
    -------
        params : list
        Tuple of parameters that guarantees the convergence.
    """
    if norm_name == None:
        # Normal case
        a = 0.4   
        w = 1.3  
        phi = -1.8
        shift = -1
        lin_params1 = (-1.3, -26)
        lin_params2 = ( 0.7, -20)
        sin_params = (a, w, phi, shift, -0.89,-20.)
        params = (*sin_params, *lin_params1, *lin_params2)
        
    elif norm_name == 'std':
    # Std_norm_case
        a = 1.9e-2  
        w = 9.7
        phi = -3.2
        shift = -1.25
        lin_params1 = (-3.2, -0.81)
        lin_params2 = ( 1.8, -4.5e-2)
        sin_params = (a, w, phi, shift, -50, -19)
        params = (*sin_params, *lin_params1, *lin_params2)
    elif norm_name == 'minmax':
    # Min_max_norm_case
        a = 0.62
        w = 32.
        phi = 12.7
        shift = 0.11
        lin_params1 = (-3., 0.23)
        lin_params2 = ( 1.7, 0.47)
        sin_params = (a, w, phi, shift, -1.31, 0.47)
        params = (*sin_params, *lin_params1, *lin_params2)
    else: raise Exception('Unknown Normalization.')
    return list(params)
