import numpy as np

class bagging_ensemble:
    """
    https://www.sciencedirect.com/science/article/pii/S000437020200190X
    """
    def __init__(self, repetition):
        self.repetition = repetition
        self.list_classifier = []
        
    def train(self, data, labels, classifier, dict_classifier, 
              dict_train, split_procedure, dict_split):
        prev = len(self.list_classifier)
        all_data = np.column_stack((data, labels))
        bootstrap_gen = self.bootstrap(all_data, self.repetition)
        for i, all_d in enumerate(bootstrap_gen):
            print(f'{i + prev}/{self.repetition + prev}')
            c = classifier(**dict_classifier)
            d = all_d[:,:-labels.shape[1]]
            l = all_d[:,-labels.shape[1]:]
            training, validation, train_labels, val_labels = split_procedure(d, l, **dict_split)
            c.train(training, train_labels, validation, val_labels, **dict_train)
            self.list_classifier.append(c)
        self.corr_matrix(data, labels)

    def predict(self, data):
        out = 0
        for c in self.list_classifier:
            out += c.predict(data)
        return out/len(self.list_classifier)
    
    def bootstrap(self, data, repetition = 20, sample_size = None):
        if sample_size == None: sample_size = len(data)
        n_subsample = np.floor(len(data)/repetition).astype(int)
        rng = np.random.default_rng()
        for _ in range(repetition):
            subsample = rng.choice(data, n_subsample, axis = 0)
            yield rng.choice(subsample, sample_size, axis = 0)
            
    @property
    def train_MEE(self):
        min_len = np.min([len(c.train_MEE) for c in self.list_classifier])
        
        train_MEE = np.array(self.list_classifier[0].train_MEE[:min_len])
        for c in self.list_classifier[1:]:
            train_MEE += np.array(c.train_MEE[:min_len])
        return train_MEE/len(self.list_classifier)
    @property
    def val_MEE(self):
        min_len = np.min([len(c.val_MEE) for c in self.list_classifier])
        val_MEE = np.array(self.list_classifier[0].val_MEE[:min_len])
        for c in self.list_classifier[1:]:
            val_MEE += np.array(c.val_MEE[:min_len])
        return val_MEE/len(self.list_classifier)

    def corr_matrix(self, data, labels):
        n = len(self.list_classifier)
        C = np.empty((labels.shape[1], n,n))
        for i, c_i in enumerate(self.list_classifier):
            for j, c_j in enumerate(self.list_classifier[:i+1]):
                res_i = c_i.predict(data) - labels
                res_j = c_j.predict(data) - labels
                for k in range(labels.shape[1]):
                    C[k, i, j] = np.sum(res_i[:,k] * res_j[:,k])
                    C[k, j, i] = C[k, i, j]
        self.C = C
        return np.squeeze(C)

    @property
    def prune_ensemble(self):
        n = len(self.list_classifier)
        fact1 = (2 * n - 1 ) * np.sum(self.C, axis = (1,2) ) 
        new_list_classifier = []
        new_C = self.C
        j = 0
        for i in range(n):
            mask = np.ones(new_C.shape[1]).astype(bool)
            mask[j] = False
            fact2 = 2 * n*n * np.sum(new_C[:,mask,j], axis = 1) + n*n * self.C[:, j, j ]
            if (fact1 > fact2).any():
                new_list_classifier.append(self.list_classifier[i])
                j = j + 1
            else:
                new_C = new_C[:, mask, :][:,:,mask]
            if j == n-1 and len(new_list_classifier) == 0:
                raise Exception('All the classifier are strongly correlated')

        if len(new_list_classifier) == 0:
            raise Exception('All the classifier are strongly correlated')
        else: 
            print(f'Pruned {n - len(new_list_classifier)}/{n} classifier')
            self.pruned_list_classifier = new_list_classifier
            self.C = new_C
