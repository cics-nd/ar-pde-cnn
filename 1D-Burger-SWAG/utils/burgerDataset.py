from torch.utils.data import DataLoader
import torch
import numpy as np

class BurgerDataset(torch.utils.data.Dataset):
    """
    Dataset for the 1D Burgers' dataset
    """
    def __init__(self, udata):
        super().__init__()
        self.u = udata

    def replaceData(self, uNew):
        '''
        Replaces training data randomly
        '''
        assert uNew.size(-1) == self.u.size(-1)
        idx = np.random.randint(0, self.u.size(0), uNew.size(0))
        self.u[idx] = uNew

    def __getitem__(self, index):
        return self.u[index]

    def __len__(self):
        return self.u.size(0)