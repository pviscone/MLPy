#%matplotlib inline
project_dir = '../../'
data_dir = project_dir + 'data/'

import sys
sys.path.append(project_dir + 'NN/')

import numpy as np
import importlib
import time
import matplotlib.pyplot as plt
import MLP
import layer
importlib.reload(MLP)
importlib.reload(layer)
from MLP import MLP
from utils.preprocessing import split,Normalize
from utils.losses import MSE, MEE

raw_data=np.loadtxt("../../data/TR.csv",delimiter=",")[:,1:]
np.random.shuffle(raw_data)

input_data=raw_data[:,1:-2]
labels=raw_data[:,-2:]

import utils.grid_search
importlib.reload(utils.grid_search)
from utils.grid_search import grid_search

#xavier initialization
def xavier(structure):
    start=np.zeros(len(structure))
    for idx,num in enumerate(structure):
        if idx==0:
            start[idx]=np.sqrt(6)/np.sqrt(structure[idx])
        else:
            start[idx]=np.sqrt(6)/np.sqrt(structure[idx-1]+structure[idx])
    return list(start)

# Structure to test: 
n_feat = np.shape(labels)[1]
model1 = {'structure':[10, n_feat], 'func':['sigmoid', 'linear'], 'starting_points': xavier([10, n_feat])}
model2 = {'structure':[15, n_feat], 'func':['sigmoid', 'linear'], 'starting_points': xavier([15, n_feat])}
model3 = {'structure':[20, n_feat], 'func':['sigmoid', 'linear'], 'starting_points': xavier([20, n_feat])}
model5=  {'structure':[4,4,n_feat], 'func':['sigmoid','sigmoid','linear'],'starting_points': xavier([4,4, n_feat])}
model4 = {'structure':[5,4,n_feat], 'func':['sigmoid','sigmoid','linear'],'starting_points': xavier([5,4, n_feat])}
model6 = {'structure':[6,6,n_feat], 'func':['sigmoid','sigmoid','linear'],'starting_points': xavier([6,6, n_feat])}
models = [model1, model2, model3, model4, model5, model6]
dict_models = {f'Model{i}': m for i, m in enumerate(models)} 

list_eta = [5e-3,1e-3]
list_alpha = np.arange(0,0.2,0.1)
list_beta = np.arange(0.5,1,0.2) ; list_lamb = [0] ; list_batch = [-1]
dict_params = {'eta':list_eta, 'alpha':list_alpha, 'lamb': list_lamb, 'epoch':[10000], 'RMSProp' : [True], 'nesterov' : [True,False] ,'batch_size' : list_batch,'beta' : list_beta , 'patience' : [100] , 'error_threshold' : [0]}

grid_results = grid_search(MLP, dict_models, dict_params, 
                           input_data, labels, MEE,
                           verbose = 1, kind = 'k_fold', k = 4)
print(dict_params)
for i in range(0,10) : print(grid_results[i])
