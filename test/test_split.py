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
        self.assertGreater(len(dataset), check_shuffled, 'Data are not shuffled!') # Returned all the data
        # testing if the returned data respect the proportions
        self.assertEqual(len(train), round(frac_train * len(dataset)), 'Proportion of train not respected' )
        self.assertEqual(len(val), round((1-frac_train) * len(dataset)), 'proportion of validationi')

if __name__ == '__main__':
    unittest.main()
    pass
