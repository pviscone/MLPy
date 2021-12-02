import unittest

package_dir = '../NN'
data_dir = '../data/'
import numpy as np
import sys
sys.path.append(package_dir)
import utils.preprocessing as pp
import importlib
import time
import matplotlib.pyplot as plt
from CC import CCNN


def test_XOR():
    input_data = [[0,0],
                  [0,1],
                  [1,0],
                  [1,1]]
    labels = [1,0,0,1]
    input_data = np.array(input_data)
    labels = np.array(labels).reshape((len(labels),1))
    hidden_func = 'sigmoid'
    output_func = 'sigmoid'
    start=1
    learning_rate=0.1
    netw = CCNN(w_init = start, hidden_actv_f = hidden_func,
                output_actv_f= output_func, eta = learning_rate)
    netw.train(input_data, labels,
           min_epoch = 10,
           stack_threshold = 0.1,
           stop_threshold = 0.1,
           max_hidden = 10,
           n_candidate = 10,
           candidate_epoch = 500)
    plt.plot(netw.train_MSE)
    plt.show()

def test_Monk_CC():
    data_monk=np.loadtxt(data_dir + "MONK/monks-1.train",usecols=range(0,7))
    labels=np.reshape(data_monk[:,0],(len(data_monk),1))
    input_data=data_monk[:,1:]
    hidden_func = 'sigmoid'
    output_func = 'sigmoid'
    start=0.1
    learning_rate=0.01
    netw = CCNN(w_init = start, hidden_actv_f = hidden_func,
                output_actv_f= output_func, eta = learning_rate)
    netw.train(input_data, labels,
               min_epoch = 500,
               stack_threshold = 0.1,
               stop_threshold = 0.1,
               max_hidden = 3,
               n_candidate = 3,
               candidate_epoch = 500)
    plt.plot(netw.train_MSE)
    plt.show()
    print(f'Epoch Trained: {netw.epoch}')
    return

if __name__ == "__main__":
    test_Monk_CC()
