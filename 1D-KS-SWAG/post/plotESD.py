'''
This script plots the time-averaged spectral energy 
density of the simulated target, AR-DenseED deterministic 
prediction and BAR-DenseED empirical mean and standard 
deviation. The produced graphic is seen in Figure 7 of 
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
from torch.optim.lr_scheduler import ReduceLROnPlateau
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

def test(args, model, test_loader, tstep=100):
    '''
    Tests the deterministic model
    Args:
        args (argparse): object with programs arguements
        model (PyTorch model): DenseED model to be tested
        test_loader (dataloader): dataloader with test cases (use createTestingLoader)
        tstep (int): number of timesteps to predict for
    Returns:
        u_out (torch.Tensor): [d x tstep x nel] predicted quantities
        u_target (torch.Tensor): [d x tstep x nel] respective target values loaded from simulator
    '''
    
    mb_size = int(len(test_loader.dataset)/len(test_loader))
    u_out = torch.zeros(len(test_loader.dataset), tstep+1, args.nel)
    u_target = torch.zeros(len(test_loader.dataset), tstep+1, args.nel)

    model.eval()

    for bidx, (input0, uTarget0) in enumerate(test_loader):
        input = input0.to(args.device)
        u_out[bidx*mb_size:(bidx+1)*mb_size,0,:] = input[:,0]
        u_target[bidx*mb_size:(bidx+1)*mb_size] = uTarget0[:,:tstep+1].cpu()

        # Auto-regress
        for t_idx in range(tstep):
            uPred = model(input[:,-2:,:])
            u_out[bidx*mb_size:(bidx+1)*mb_size,t_idx+1,:] = uPred[:,0].cpu()
            
            input = input[:,-4:,:].detach()
            input0 = uPred[:,0,:].unsqueeze(1).detach()
            input = torch.cat([input,  input0], dim=1)

    return u_out, u_target


def testSample(args, swag_nn, test_loader, tstep=100, n_samples=10):
    '''
    Tests the samples of the Bayesian SWAG model
    Args:
        args (argparse): object with programs arguements
        model (PyTorch model): DenseED model to be tested
        test_loader (dataloader): dataloader with test cases (use createTestingLoader)
        tstep (int): number of timesteps to predict for
        n_samples (int): number of model samples to draw
    Returns:
        u_out (torch.Tensor): [d x nsamples x tstep x nel] predicted quantities of each sample
        u_target (torch.Tensor): [d x tstep x nel] respective target values loaded from simulator
    '''
    
    mb_size = int(len(test_loader.dataset)/len(test_loader))
    u_out = torch.zeros(len(test_loader.dataset), n_samples, tstep+1, args.nel)
    u_target = torch.zeros(len(test_loader.dataset), tstep+1, args.nel)

    for i in range(n_samples):
        print('Executing model sample {:d}'.format(i))
        model = swag_nn.sample(diagCov=True)
        model.eval()

        for bidx, (input0, uTarget0) in enumerate(test_loader):
            input = input0.to(args.device)
            u_out[bidx*mb_size:(bidx+1)*mb_size, i, 0, :] = input[:,0]
            if(i == 0):
                u_target[bidx*mb_size:(bidx+1)*mb_size] = uTarget0[:,:tstep+1].cpu()
            # Auto-regress
            for tidx in range(tstep):
                uPred = model(input[:,-2:,:])
                u_out[bidx*mb_size:(bidx+1)*mb_size, i, tidx+1] = uPred[:,0].detach().cpu()
                
                input = input[:,-4:,:].detach()
                input0 = uPred[:,0,:].unsqueeze(1).detach()
                input = torch.cat([input,  input0], dim=1)

    return u_out, u_target

def plotESDPlot(uPred, uPredBayes, uTarget, case=0, dx=1.0):
    '''
    Plots the energy spectral density plot
    Args:
        uPred (np.array): deterministic prediction
        uPredBayes (np.array): prediction samples from Bayesian model
        uTarget (np.array): target response field
        cases (list): list of test cases indexes
        dx (int): domain discretization size
    '''
    plt.close("all")
    # Create figure
    mpl.rcParams['font.family'] = ['serif'] # default is sans-serif
    rc('text', usetex=True)

    fig = plt.figure(figsize=(7, 5))
    ax = []
    axins=[]
    ax.append(plt.subplot2grid((1, 1), (0, 0), colspan=1))
    axins.append(zoomed_inset_axes(ax[0], 2.0, loc=3)) # zoom-factor: 2.5, location: upper-left

    cmap = mpl.cm.get_cmap('Set1')

    uPred = uPred[case]
    uPredBayes = uPredBayes[case]
    uTarget = uTarget[case]

    # === Target ESD ===
    # FFT and normalize by length of the series
    # https://cran.r-project.org/web/packages/psd/vignettes/normalization.pdf
    ps = np.abs(np.fft.rfft(uTarget))**2/uTarget.shape[-1]
    freqs = np.fft.rfftfreq(uTarget.shape[1], dx)
    idx = np.argsort(freqs)
    # Average in time and plot
    ax[0].plot(freqs[idx[1:-1]], np.average(ps[:,idx[1:-1]], axis=0), '-+', c='k', label="Target")
    axins[0].plot(freqs[idx[1:-1]], np.average(ps[:,idx[1:-1]], axis=0), '-+', c='k')

    # === Deterministic ESD ===
    # FFT and normalize by length of the series
    ps = np.abs(np.fft.rfft(uPred))**2/uPred.shape[-1]
    freqs = np.fft.rfftfreq(uPred.shape[-1], dx)
    idx = np.argsort(freqs) # Sort frequencies for plotting
    # Average in time and plot
    ax[0].plot(freqs[idx[1:-1]], np.average(ps[:,idx[1:-1]], axis=0), '-+', c='r', label="AR-DenseED", alpha=0.7)
    axins[0].plot(freqs[idx[1:-1]], np.average(ps[:,idx[1:-1]], axis=0), '-+', c='r', alpha=0.7)

    # === Bayesian ESD ===
    freq_samples = []
    for i in range(uPredBayes.shape[0]):
        uPred0 = uPredBayes[i] # Transpose so we take fft in the spacial domain
        # FFT and normalize by length of the series
        ps = np.abs(np.fft.rfft(uPred0))**2/uPred0.shape[-1]
        freqs = np.fft.rfftfreq(uPred0.shape[-1], dx)
        idx = np.argsort(freqs) # Sort frequencies for plotting

        freq_samples.append(np.average(ps[:,idx[1:-1]], axis=0))

    # Plot mean and 2 sigma
    freq_samples = np.stack(freq_samples, axis=0)
    freq_mean = np.average(freq_samples, axis=0)
    freq_std = np.std(freq_samples, axis=0) 
    ax[0].plot(freqs[idx[1:-1]], freq_mean, '-+', c='b', linewidth=2.0, markersize=7, label='BAR-DenseED Mean', alpha=0.7)
    ax[0].plot(freqs[idx[1:-1]], freq_mean+2*freq_std, '-', c='b', linewidth=1.0, zorder=1, alpha=0.5)
    ax[0].plot(freqs[idx[1:-1]],  np.abs(freq_mean-2*freq_std), '-', c='b', linewidth=1.0, zorder=1, alpha=0.5)
    ax[0].fill_between(freqs[idx[1:-1]], freq_mean+2*freq_std,  np.abs(freq_mean-2*freq_std), facecolor='b', alpha=0.2, label=r'BAR-DenseED $2\sigma$')
    
    # Zoomed in plot
    axins[0].plot(freqs[idx[1:-1]], freq_mean, '-+', c='b', linewidth=2.0, markersize=7, alpha=0.7)
    axins[0].plot(freqs[idx[1:-1]], freq_mean+2*freq_std, '-', c='b', linewidth=1.0, zorder=1, alpha=0.5)
    axins[0].plot(freqs[idx[1:-1]], np.abs(freq_mean-2*freq_std), '-', c='b', linewidth=1.0, zorder=1, alpha=0.5)
    axins[0].fill_between(freqs[idx[1:-1]], freq_mean+2*freq_std,  np.abs(freq_mean-2*freq_std), facecolor='b', alpha=0.2)

    # Axis parameters
    ax[0].set_yscale('log')
    ax[0].set_xlabel('Hz', fontsize=14)
    ax[0].set_ylabel('E(k)/Hz', fontsize=14)
    ax[0].legend(prop={'size': 14}, loc='upper right')

    mark_inset(ax[0], axins[0], loc1=2, loc2=4, fc="none", ec="0.5", zorder=-1)
    for axin0 in axins:
        axin0.set_yscale('log')
        axin0.set_xlim([0, 0.25]) # apply the x-limits
        axin0.set_ylim([0.1, 50]) # apply the y-limits
        axin0.set_xticklabels([])
        axin0.set_yticklabels([])

    del(uPred)
    gc.collect()
    file_dir = '.'
    # If directory does not exist create it
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)
    file_name = file_dir+"/ks_ESD"
    plt.savefig(file_name+".png")
    plt.savefig(file_name+".pdf")

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

    # Create testing loaders
    ksLoader = KSLoader()
    test_cases = np.arange(196,200+1e-8,1).astype(int)
    testing_loader = ksLoader.createTestingLoader('../solver', test_cases, dt=args.dt, batch_size=5, tmax=5001)

    # Create DenseED model
    denseED = DenseED(in_channels=2, out_channels=1,
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

    nsteps = 5000
    # Deterministic model
    swag_nn.loadModel(100, file_dir='networks')
    with torch.no_grad():
        uPred, uTarget = test(args, swag_nn.base, testing_loader, tstep=nsteps)
    # Bayesian model samples
    swag_nn.loadModel(200, file_dir='networks')
    with torch.no_grad():
        uPredBayes, uTarget = testSample(args, swag_nn, testing_loader, tstep=nsteps, n_samples=30)
    
    plotESDPlot(uPred.detach().numpy(), uPredBayes.detach().numpy(), uTarget.detach().numpy(), case=0, dx=args.dx)