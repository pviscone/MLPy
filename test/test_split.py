import unittest

package_dir = '../NN'
data_dir = '../data/'
import numpy as np
import sys
sys.path.append(package_dir)
import utils.preprocessing as pp

class TestSplit(unittest.TestCase):

    def test_hold_out(self):
        dataset = np.arange(100).reshape((10,10))
        frac_train = 0.56
        train, val = pp.split(dataset, frac_training = frac_train, shuffle = True)
        data_list = dataset.tolist()
        check_all_in = 0
        check_shuffled = 0

        # Testing the shuffle
        for i, t in enumerate(train):
            check_all_in += (list(t) in data_list) # boolean add (+1 or +0)
            check_shuffled += (list(t) == data_list[i]) # // // //
        for j, v in enumerate(val, start = i):
            check_all_in += (list(v) in data_list) # boolean add (+1 or +0)
            check_shuffled += (list(v) == data_list[j]) # // // //

        self.assertEqual(check_all_in, len(dataset), 'Missing some data!') # Returned all the data
        self.assertGreater(len(dataset), check_shuffled, 'Data are not shuffled!')
        self.assertEqual(len(train), round(frac_train * len(dataset)), 'Proportion of train not respected' )
        self.assertEqual(len(val), round((1-frac_train) * len(dataset)), 'proportion of validation not respected')

    def test_kfold(self):
        k = 3
        dataset = np.arange(20).reshape((10,2))
        train, idxs = pp.split(dataset, kind = 'k-fold', k = k, shuffle = False)
        # To remember who to split with indexs of idxs...
        #for idx1, idx2 in idxs:
        #    print('train set')
        #    print(np.delete(train, slice(idx1, idx2), axis = 0))
        #    print('validation set')
        #    print(train[idx1:idx2])
        #    print('\n')
        self.assertEqual(len(idxs), k, 'Returned wrong number of splits') # Just a stupid test

if __name__ == '__main__':
    unittest.main()
    pass
