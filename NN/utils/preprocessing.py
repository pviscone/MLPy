import numpy as np

def split(input_matrix,frac_training=0.8,shuffle=False,kind="hold_out",k=4):
    """[Splitting the data in three different set, one for training, one for vali
	dation, one for test. Each pattern of each set is selected randomly from
	the initial input_matrix.]

    Args:
        input_matrix ([type]): [input matrix without test set]
        frac_training (float, optional): [fraction of data to use in training]. Defaults to 0.8.
        shuffle (bool, optional): [shuffle the input matrix]. Defaults to False.
    """
    copy_data = np.copy(input_matrix) # Copy the dataset to not change original
                                      # input_matrix
    if shuffle:
        np.random.shuffle(copy_data)
    if kind=="hold_out":
        idx=round(len(copy_data)*frac_training)
        training_set=copy_data[0:idx,:]
        validation_set=copy_data[idx:,:]
        return training_set, validation_set
    elif kind=="k-fold":
        pass
