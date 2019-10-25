'''
This script plots the average mean squared error (MSE) and 
energy squared error (ESE) as a function of time for a test 
set of 200 cases using AR-DenseED and predictive expectation 
of BAR-DenseED. The produced graphic is seen in Figure 6 of 
the paper.
===
Distributed by: Notre Dame CICS (MIT Liscense)
- Associated publication:
url: http://www.sciencedirect.com/science/article/pii/S0021999119307612
doi: https://doi.org/10.1016/j.jcp.2019.109056
github: https://github.com/cics-nd/ar-pde-cnn
===
'''
import sys
sys.path.append("..") # Adds higher directory to python modules path.

from args import Parser
# from models.denseED import DenseED
from nn.denseEDcirc import DenseED
from nn.bayesNN import BayesNN
from nn.swag import SwagNN
from nn.ksFiniteDifference import KSIntegrate
from utils.utils import mkdirs
from utils.ksLoader import KSLoader

from torch.autograd import Variable
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader, TensorDataset
import torch
import gc

import matplotlib as mpl
import numpy as np
import os, time
import scipy.integrate as integrate
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.lines import Line2D
from matplotlib import rc
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset

def testMSE(args, model, test_loader, tstep=1000):
    '''
    Tests the deterministic model and calculates the mean squared error
    Args:
        args (argparse): object with programs arguements
        model (PyTorch model): DenseED model to be tested
        test_loader (dataloader): dataloader with test cases (use createTestingLoader)
        tstep (int): number of timesteps to predict for
    Returns:
        mse_error (torch.Tensor): [d x tstep] mean squared error over the domain
    '''

    model.eval()
    mb_size = int(len(test_loader.dataset)/len(test_loader))
    u_out = torch.zeros(len(test_loader.dataset), tstep+1, args.nel)
    u_target = torch.zeros(len(test_loader.dataset), tstep+1, args.nel)

    for bidx, (input0, uTarget0) in enumerate(test_loader):
        input = input0.to(args.device)
        u_target[bidx*mb_size:(bidx+1)*mb_size] = uTarget0[:,:tstep+1]
        u_out[bidx*mb_size:(bidx+1)*mb_size,0, :] = input[:,0]
        # Auto-regress
        for t_idx in range(tstep):
            uPred = model(input[:,-args.nic:,:])
            u_out[bidx*mb_size:(bidx+1)*mb_size,t_idx+1,:] = uPred[:,0]
            
            input = input[:,-(args.nic-1):,:].detach()
            input0 = uPred[:,0,:].unsqueeze(1).detach()
            input = torch.cat([input,  input0], dim=1)


    mse_error = torch.mean(torch.pow(u_out - u_target,2), dim=-1)

    return mse_error

def testSamplesMSE(args, model, test_loader, tstep=1000, n_samples=10):
    '''
    Tests the samples of the Bayesian SWAG model and calculates the mean
    squared error using the expected value
    Args:
        args (argparse): object with programs arguements
        model (PyTorch model): DenseED model to be tested
        test_loader (dataloader): dataloader with test cases (use createTestingLoader)
        tstep (int): number of timesteps to predict for
        n_samples (int): number of model samples to draw
    Returns:
        mse_error (torch.Tensor): [d x tstep] mean squared error over the domain
    '''

    mb_size = int(len(test_loader.dataset)/len(test_loader))
    u_out = torch.zeros(len(test_loader.dataset), n_samples, tstep+1, args.nel)
    u_target = torch.zeros(len(test_loader.dataset), tstep+1, args.nel)
    
    for i in range(n_samples):
        print('Executing model sample {:d}'.format(i))
        model = swag_nn.sample(diagCov=False)
        model.eval()

        for bidx, (input0, uTarget0) in enumerate(test_loader):
            input = input0.to(args.device)
            u_target[bidx*mb_size:(bidx+1)*mb_size] = uTarget0[:,:tstep+1]
            u_out[bidx*mb_size:(bidx+1)*mb_size, i, 0, :] = input[:,0]
            # Auto-regress
            for t_idx in range(tstep):
                uPred = model(input[:,-args.nic:,:])
                u_out[bidx*mb_size:(bidx+1)*mb_size, i, t_idx+1,:] = uPred[:,0]
                
                input = input[:,-(args.nic-1):,:].detach()
                input0 = uPred[:,0,:].unsqueeze(1).detach()
                input = torch.cat([input,  input0], dim=1)


    mse_error = torch.mean(torch.pow(torch.mean(u_out, dim=1) - u_target,2), dim=-1)

    return mse_error

def plotError(tT, mse_errors, mse_errors_bayes):
    '''
    Creates grid of 4 different test cases, plots target, prediction, variance and error for each
    '''
    plt.close("all")
    # Create figure
    mpl.rcParams['font.family'] = ['serif'] # default is sans-serif
    rc('text', usetex=True)

    fig = plt.figure(figsize=(8, 4))
    ax = []
    ax.append(plt.subplot2grid((1, 1), (0, 0), colspan=1))

    ax[0].plot(tT, np.mean(mse_errors, axis=0), '-', c='r', label='AR-DenseED')
    ax[0].plot(tT, np.mean(mse_errors_bayes, axis=0), '--', c='b', label='BAR-DenseED')

    ax[0].set_xlabel('t', fontsize=14)
    ax[0].set_ylabel('MSE', fontsize=14)
    ax[0].legend(loc='upper left')

    file_dir = '.'
    # If directory does not exist create it
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)
    file_name = file_dir+"/ks_MSE"
    plt.savefig(file_name+".png", bbox_inches='tight')
    plt.savefig(file_name+".pdf", bbox_inches='tight')
    plt.show()

if __name__ == '__main__':

    # Parse arguements
    args = Parser().parse(dirs=False)
    use_cuda = "cpu"
    if(torch.cuda.is_available()):
        use_cuda = "cuda"
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Torch device:{}".format(args.device))
    
    # Timestep
    tsteps = 200
    dt = 0.1
    # Number training per timestep
    epochs = 200
    batch_size = 64

    # Domain settings
    x0 = 0
    x1 =  22*np.pi
    args.dx = (x1 - x0)/args.nel

    nsteps = 1000

    # Create testing loaders
    ksLoader = KSLoader()
    test_cases = np.arange(1,200+1e-8,1).astype(int)
    testing_loader = ksLoader.createTestingLoader('../solver', test_cases, dt=args.dt, tmax=nsteps, batch_size=50)

    # Create DenseED model
    denseED = DenseED(in_channels=args.nic, out_channels=args.noc,
                        blocks=args.blocks,
                        growth_rate=args.growth_rate, 
                        init_features=args.init_features,
                        bn_size=args.bn_size,
                        drop_rate=args.drop_rate,
                        bottleneck=False,
                        out_activation=None).to(args.device)

    # Bayesian neural network
    bayes_nn = BayesNN(args, denseED)
    swag_nn = SwagNN(args, bayes_nn, full_cov=True, max_models=10)
    
    # First predict with determinisitic
    swag_nn.loadModel(100, file_dir='networks')
    with torch.no_grad():
        mse_error = testMSE(args, swag_nn.base, testing_loader, tstep=nsteps)

    # Predict with Bayesian
    swag_nn.loadModel(200, file_dir='networks')
    with torch.no_grad():
        mse_error_bayes = testSamplesMSE(args, swag_nn, testing_loader, tstep=nsteps, n_samples=30)
    

    tT = np.arange(0, nsteps*args.dt+1e-8, args.dt)
    plotError(tT, mse_error.cpu().numpy(), mse_error_bayes.cpu().numpy())
