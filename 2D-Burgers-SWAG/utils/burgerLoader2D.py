from torch.utils.data import DataLoader, TensorDataset

import torch
import numpy as np
import os

class BurgerLoader():
    '''
    Class used for creating data loaders for the 2d burgers equation
    Args:
        dt (float): time-step size of the model
        shuffle (boolean): shuffle the training data or not
    '''
    def __init__(self, dt=0.01, shuffle=True):

        self.dt = dt
        self.shuffle = shuffle

    def createTrainingLoader(self, ncases, nel, batch_size=64):
        '''
        Loads in training data from Fenics simulator
        Args:
            data_dir (string): directory of data
            ncases (int): number of training cases to use
            n_init (int): number of intial conditions to use from each case
            batch_size (int): mini-batch size
        '''
        # Create on the fly Dataset
        dataSet = InitPeriodicCond2d(nel, ncases)

        training_loader = DataLoader( dataSet,
            batch_size=batch_size, shuffle=self.shuffle, drop_last=True)
        # Save original training loader
        self.training_loader0 = training_loader
        return training_loader

    def createTestingLoader(self, data_dir, cases, tMax=1.0, simdt=0.001, save_every=2, batch_size=1):
        '''
        Loads in training data from Fenics simulator, assumes simulator has saved
        each time-step at specified delta t
        Args:
            data_dir (string): directory of data
            cases (np.array): array of training cases, must be integers
            tMax (float): maximum time value to load simulator data up to
            simdt (float): time-step size used in the simulation
            save_every (int): Interval to load the training data at (default is 2 to match FEM simulator)
            batch_size (int): mini-batch size
        Returns:
            test_loader (Pytorch DataLoader): Returns testing loader
        '''
        testing_data = []
        target_data = []

        # Loop through test cases
        for i, val in enumerate(cases):
            case_dir = os.path.join(data_dir, "run{:d}".format(val))
            print("Reading test case: {}".format(case_dir))
            seq = []
            for j in range(0, int(tMax/simdt)+1, save_every):
                file_dir = os.path.join(case_dir, "u{:d}.npy".format(j))
                u0 = np.load(file_dir)
                # Remove the periodic nodes
                seq.append(u0[:,:,:])

            file_dir = os.path.join(case_dir, "u0.npy")
            uInit = np.load(file_dir)
            uTarget = np.stack(seq, axis=0)

            # Remove the periodic nodes and unsqueeze first dim
            testing_data.append(torch.Tensor(uInit[:,:,:]).unsqueeze(0))
            target_data.append(torch.Tensor(uTarget))
        # Create data loader
        data_tuple = (torch.cat(testing_data, dim=0), torch.stack(target_data, dim=0))
        testing_loader = DataLoader(TensorDataset(*data_tuple),
            batch_size=batch_size, shuffle=False, drop_last=False)

        return testing_loader


class InitPeriodicCond2d(torch.utils.data.Dataset):
    """Generate periodic initial condition on the fly.

    Args:
        order (int): order of Fourier series expansion
        ncells (int): spatial discretization over [0, 1]
        nsamples (int): total # samples
    """
    def __init__(self, ncells, nsamples, order=4):
        super().__init__()
        self.order = order
        self.nsamples = nsamples
        self.ncells = ncells
        x = np.linspace(0, 1, ncells+1)[:-1]
        xx, yy = np.meshgrid(x, x)
        aa, bb = np.meshgrid(np.arange(-order, order+1), np.arange(-order, order+1))
        k = np.stack((aa.flatten(), bb.flatten()), 1)
        self.kx_plus_ly = (np.outer(k[:, 0], xx.flatten()) + np.outer(k[:, 1], yy.flatten()))*2*np.pi
 
    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            init_condition
        """
        np.random.seed(index+100000) # Make sure this is different than the seeds set in finite element solver!
        lam = np.random.randn(2, 2, (2*self.order+1)**2)
        c = -1 + np.random.rand(2) * 2

        f = np.dot(lam[0], np.cos(self.kx_plus_ly)) + np.dot(lam[1], np.sin(self.kx_plus_ly))
        f = 2 * f / np.amax(np.abs(f), axis=1, keepdims=True) + c[:, None]
        return torch.FloatTensor(f).reshape(-1, self.ncells, self.ncells)

    def __len__(self):
        # generate on-the-fly
        return self.nsamples