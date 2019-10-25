from utils.burgerDataset import BurgerDataset
from torch.utils.data import DataLoader, TensorDataset

import torch
import numpy as np

class BurgerLoader():
    '''
    Class used for creating data loaders for the burgers equation
    Args:
        dt (float): time-step size of the model
        shuffle (boolean): shuffle the training data or not
    '''
    def __init__(self, dt=0.01, shuffle=True):

        self.dt = dt
        self.shuffle = shuffle

    def createTrainingLoader(self, data_dir, cases, n_init, t_range=[0,2], dt=0.001, batch_size=64):
        '''
        Loads in training data from Fenics simulator
        Args:
            data_dir (string): directory of data
            cases (np.array): array of training cases, must be integers
            n_init (int): number of intial conditions to use from each case
            batch_size (int): mini-batch size
        '''
        training_data = []

        # Indexes of data time-steps to use as intial conditions
        t_start = int(t_range[0]/dt)
        t_end = int(t_range[1]/dt)
        nidx = np.linspace(t_start, t_end, n_init).astype(int)

        for i, val in enumerate(cases):
            file_name = data_dir+"/u{:d}.npy".format(val)
            print("Reading file: {}".format(file_name))
            u = np.load(file_name)
            # Convert to tensor and unsqueeze channel dim
            # Remove last element due to periodic conditions between [0,1]
            uTensor = torch.Tensor(u[nidx, :-1]).unsqueeze(1)
            training_data.append(uTensor)

        self.training_data = BurgerDataset(torch.cat(training_data, dim=0))

        training_loader = DataLoader( self.training_data,
            batch_size=batch_size, shuffle=self.shuffle, drop_last=True)

        return training_loader

    def createTrainingLoaderInitial(self, ncases, x0, x1, nel, batch_size=64):
        '''
        Creates initial states using truncated Fourier series
        Args:
            ncases (int): number of training 
            x0 (float): start of domain
            x1 (float): end of domain
            nel (int): number of elements to discretize the domain by
            batch_size (int): mini-batch size
        '''
        training_data = []

        order=3
        x = np.linspace(x0, x1, nel+1)

        for i in range(ncases):

            lam = np.random.randn(2, 2*order+1)
            c = np.random.rand() - 0.5
            k = np.arange(-order, order+1)
            kx = np.outer(k, x)*2*np.pi

            # vector field
            f = np.dot(lam[[0]], np.cos(kx)) + np.dot(lam[[1]], np.sin(kx))
            f = 2 * f / np.amax(np.abs(f)) + 2*c

            uTensor = torch.Tensor(f[:,:-1]).unsqueeze(0)
            training_data.append(uTensor.repeat(1,20,1))

        self.training_data = BurgerDataset(torch.cat(training_data, dim=0))
        training_loader = DataLoader( self.training_data,
            batch_size=batch_size, shuffle=self.shuffle, drop_last=True)

        return training_loader

    def createTestingLoader(self, data_dir, cases, t_start=0, dt=0.001, batch_size=1):
        '''
        Loads in training data from Fenics simulator
        Args:
            data_dir (string): directory of data
            cases (np.array): array of training cases, must be integers
            n_init (int): number of intial conditions to use from each case
            batch_size (int): mini-batch size
        '''
        testing_data = []
        target_data = []

        # Indexes of data time-steps to use as intial conditions
        nidx = int(t_start/dt)

        for i, val in enumerate(cases):
            file_name = data_dir+"/u{:d}.npy".format(val)
            print("Reading file: {}".format(file_name))
            u = np.load(file_name)

            # Convert to tensor and unsqueeze channel dim
            uTensor = torch.Tensor(u[nidx, :-1]).unsqueeze(0).unsqueeze(1)
            testing_data.append(uTensor.repeat(1,20,1))
            # Remove last element due to periodic conditions between [0,1]
            target_data.append(torch.Tensor(u[::int(self.dt/dt),:-1]))

        data_tuple = (torch.cat(testing_data, dim=0), torch.stack(target_data, dim=0))
        testing_loader = DataLoader(TensorDataset(*data_tuple),
            batch_size=batch_size, shuffle=False, drop_last=False)

        return testing_loader
