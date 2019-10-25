from torch.utils.data import DataLoader, TensorDataset
import torch
import numpy as np

class KSLoader():
    '''
    Class used for creating data loaders for the burgers equation
    '''
    def __init__(self, shuffle=True):
        self.shuffle = shuffle
    
    def createTrainingLoader(self, data_dir, cases, n_init, dt=0.1, batch_size=64):
        '''
        Loads in training data from matlab simulator
        Args:
            data_dir (string): directory of data
            cases (np.array): array of training cases, must be integers
            n_init (int): number of intial conditions to use from each case
            dt (float): time-step of matlabe data
            batch_size (int): mini-batch size
        '''
        training_data =[]
        for i, val in enumerate(cases):
            file_name = data_dir+"/ks_data_{:d}.dat".format(val)
            print("Reading file: {}".format(file_name))
            u = np.loadtxt(file_name, delimiter=',')
            u = u[:,:-1]
            
            nidx = np.linspace(100/dt, u.shape[0]-1, n_init).astype(int)
            uTensor = torch.Tensor(u[nidx, :]).unsqueeze(1)
            uTensor = uTensor.repeat(1,5,1)
            
            training_data.append(uTensor)

        training_loader = DataLoader(torch.cat(training_data, dim=0),
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
        x = np.linspace(x0, x1, nel+1)

        for i in range(ncases):

            a = np.random.rand()+2.5 # Amplitude of waves
            k = np.linspace(1, 3, 3) # Fourier series order
            
            # https://epubs.siam.org/doi/pdf/10.1137/070705623
            # The mean most unstable mode |k| = L/(2*pi*sqrt(2)),
            mu_n = int((x1-x0)/(2*np.pi*np.sqrt(2)))
            n = np.round(np.abs(0.5*np.random.randn()+mu_n)) # Mode number
            l = x1/(2*n) # Wave-length

            c = 2*np.pi*np.random.rand() # Off set
            lam0 = 2*np.random.randn() # Pertabation
            lam = np.ones(k.shape[0])
            lam[1] = lam0
            lam = lam/k

            kx = np.outer(k, x)*np.pi/l + c

            # vector field
            f = np.sum(np.tile(lam[:,np.newaxis], (kx.shape[1]))*np.sin(kx), axis=0)
            # Set wave amplitude to a
            f = 2*a*(f - np.amin(f)) / (np.amax(f) - np.amin(f)) - a

            uTensor = torch.Tensor(f[:-1]).unsqueeze(0).unsqueeze(0)
            uTensor = uTensor.repeat(1,5,1)
            training_data.append(uTensor)

        training_loader = DataLoader( torch.cat(training_data, dim=0),
            batch_size=batch_size, shuffle=self.shuffle, drop_last=True)

        return training_loader


    def createTestingLoader(self, data_dir, cases, dt = 0.1, tmax=1000, batch_size=64):
        '''
        Loads in testing data from matlab simulator; includes target values in dataloader
        Args:
            data_dir (string): directory of data
            cases (np.array): array of training cases, must be integers
            n_init (int): number of intial conditions to use from each case
            batch_size (int): mini-batch size
        '''
        testing_data = []
        target_data = []
        for i, val in enumerate(cases):
            file_name = data_dir+"/ks_data_{:d}.dat".format(val)
            print("Reading file: {}".format(file_name))
            u = np.loadtxt(file_name, delimiter=',')
            u = u[:,:-1]
            # Initial state
            uTensor = torch.Tensor(u[int(100/dt), :]).unsqueeze(0).unsqueeze(0)
            testing_data.append(uTensor.repeat(1,5,1))
            # Full target field
            target_data.append(torch.Tensor(u[int(100/dt):int(100/dt)+tmax+1, :]))

        data_tuple = (torch.cat(testing_data, dim=0), torch.stack(target_data, dim=0))

        testing_data = DataLoader(TensorDataset(*data_tuple),
            batch_size=batch_size, shuffle=False)

        return testing_data