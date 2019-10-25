'''
This script plots the prediction of the auto-regressive DenseED
for three various test cases seen in Figure 5 of the paper.
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
from torch.optim.lr_scheduler import ReduceLROnPlateau, ExponentialLR

import torch
import torch.nn.functional as F

import matplotlib as mpl
import numpy as np
import os
import scipy.integrate as integrate
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.lines import Line2D
from matplotlib import rc


def test(args, model, test_loader, mb_size=5, tstep=100):
    '''
    Tests the model
    Args:
        model (PyTorch model): DenseED model to be tested
        device (PtTorch device): device model is on
        test_loader (dataloader): dataloader with test cases (use createTestingLoader)
        mb_size (int): test mini-batch size
        tstep (int): number of timesteps to predict for
    Returns:
        uPred (torch.Tensor): [t x n] predicted quantities
    '''
    mb_size = int(len(test_loader.dataset)/len(test_loader))
    u_out = torch.zeros(len(test_loader.dataset), tstep+1, args.nel)
    u_target = torch.zeros(len(test_loader.dataset), tstep+1, args.nel)

    model.eval()

    for bidx, (input0, uTarget0) in enumerate(test_loader):
        input = input0.to(args.device)
        u_out[bidx*mb_size:(bidx+1)*mb_size,0,:] = input[:,0]
        u_target[bidx*mb_size:(bidx+1)*mb_size] = uTarget0.cpu()

        # Auto-regress
        for t_idx in range(tstep):
            uPred = model(input[:,-2:,:])
            u_out[bidx*mb_size:(bidx+1)*mb_size,t_idx+1,:] = uPred[:,0].cpu()
            
            input = input[:,-4:,:].detach()
            input0 = uPred[:,0,:].unsqueeze(1).detach()
            input = torch.cat([input,  input0], dim=1)

    return u_out, u_target

def plotPred(t, xT, uPred, uTarget, ncases=3):
    '''
    Plots prediction contour
    '''
    plt.close("all")
    # Create figure
    mpl.rcParams['font.family'] = ['serif'] # default is sans-serif
    rc('text', usetex=True)

    fig = plt.figure(figsize=(16, 10))
    ax = [[] for i in range(ncases)]
    for i in range(ncases):
        ax[i].append(plt.subplot2grid((3, 7*ncases), (0, 7*i), colspan=6))
        ax[i].append(plt.subplot2grid((3, 7*ncases), (1, 7*i), colspan=6))
        ax[i].append(plt.subplot2grid((3, 7*ncases), (2, 7*i), colspan=6))

    ax = np.array(ax).T

    plotTestCase(fig, ax[:,0], t, xT, uPred[0].T, uTarget[0].T, cbar=False)
    plotTestCase(fig, ax[:,1], t, xT, uPred[1].T, uTarget[1].T, cbar=False)
    plotTestCase(fig, ax[:,2], t, xT, uPred[2].T, uTarget[2].T, cbar=True)

    file_dir = '.'
    # If directory does not exist create it
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)

    file_name = file_dir+"/ks_AR_Pred"
    plt.savefig(file_name+".png", bbox_inches='tight')
    plt.savefig(file_name+".pdf", bbox_inches='tight')
    plt.show()

def plotTestCase(fig, ax, t, xT, uPred, uTarget, cbar=True):
    '''
    Plots specific test case
    '''
    cmap = "rainbow"
    T, X = np.meshgrid(t + 0.1, xT)

    # Target
    c0 = ax[0].imshow(uTarget[:t.shape[0],:], interpolation='nearest', cmap=cmap, origin='lower', aspect='auto', extent=[t[0],t[-1],xT[0],xT[-1]])
    c0.set_clim(vmin=-3, vmax=3)

    # Prediction
    c0 = ax[1].imshow(uPred, interpolation='nearest', cmap=cmap, origin='lower', aspect='auto', extent=[t[0],t[-1],xT[0],xT[-1]])
    c0.set_clim(vmin=-3, vmax=3)

    if(cbar):
        p0 = ax[0].get_position().get_points().flatten()
        p1 = ax[1].get_position().get_points().flatten()
        ax_cbar = fig.add_axes([p1[2]+0.02, p1[1], 0.020, p0[3]-p1[1]])
        ticks = np.linspace(0, 1, 5)
        tickLabels = np.linspace(-3, 3, 5)
        tickLabels = ["{:02.2f}".format(t0) for t0 in tickLabels]
        cbar = mpl.colorbar.ColorbarBase(ax_cbar, cmap=plt.get_cmap(cmap), orientation='vertical', ticks=ticks)
        cbar.set_ticklabels(tickLabels)

    # Error
    cmap = "viridis"
    c0 = ax[2].imshow( np.abs(uTarget - uPred), interpolation='nearest', cmap=cmap, origin='lower', aspect='auto', extent=[t[0],t[-1],xT[0],xT[-1]])
    c0.set_clim(vmin=0, vmax=5.5)
    if(cbar):
        p0 = ax[2].get_position().get_points().flatten()
        ax_cbar = fig.add_axes([p0[2]+0.02, p0[1], 0.020, p0[3]-p0[1]])
        ticks = np.linspace(0, 1, 5)
        tickLabels = np.linspace(c0.norm.vmin, c0.norm.vmax, 5)
        tickLabels = ["{:02.2f}".format(t0) for t0 in tickLabels]
        cbar = mpl.colorbar.ColorbarBase(ax_cbar, cmap=plt.get_cmap(cmap), orientation='vertical', ticks=ticks)
        cbar.set_ticklabels(tickLabels)

    ax[0].set_xticklabels('', fontsize=14)
    ax[1].set_xticklabels('', fontsize=14)
    ax[0].set_ylabel('x', fontsize=14)
    ax[1].set_ylabel('x', fontsize=14)
    ax[2].set_ylabel('x', fontsize=14)
    ax[2].set_xlabel('t', fontsize=14)


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
    nel = 96
    x0 = 0
    x1 =  22*np.pi
    deltaX = (x1 - x0)/nel

    # Create testing loaders
    ksLoader = KSLoader()
    test_cases = np.arange(196,200+1e-8,1).astype(int)
    testing_loader = ksLoader.createTestingLoader('../solver', test_cases, dt=args.dt, batch_size=5)

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
    

    swag_nn.loadModel(100, file_dir='networks')

    uPred, uTarget = test(args, swag_nn.base, testing_loader, tstep=1000)
    
    tTest = np.arange(0, (1000)*dt+1e-8, dt)
    xTest = np.linspace(x0, x1, nel)

    plotPred(tTest, xTest, uPred.detach().numpy(), uTarget.detach().numpy())

