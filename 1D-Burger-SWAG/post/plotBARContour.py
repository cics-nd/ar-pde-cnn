'''
This script plots the four test predictions of the predictive
expectation and variance of BAR-DenseED seen in Figure 13 of the paper.
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
from nn.denseEDcirc import DenseED
from nn.bayesNN import BayesNN
from nn.swag import SwagNN
from utils.utils import mkdirs
from utils.burgerLoader import BurgerLoader

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib import rc
import matplotlib.gridspec as gridspec

import torch
import numpy as np
import os
import time

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
    u_out = torch.zeros(mb_size, n_samples, tstep+1, args.nel)
    betas = torch.zeros(n_samples)
    
    for i in range(n_samples):
        print('Executing model sample {:d}'.format(i))
        model = swag_nn.sample(diagCov=True)
        model.eval()
        betas[i] = model.model.log_beta.exp()
        for batch_idx, (input0, uTarget0) in enumerate(test_loader):
            input = input0.to(args.device)
            u_target = uTarget0
            u_out[:,i,0,:] = input[:,0]
            # Auto-regress
            for t_idx in range(tstep):
                uPred = model(input[:,-args.nic:,:])

                u_out[:,i,t_idx+1,:] = uPred[:,0]

                input = input[:,-int(args.nic-1):,:].detach()
                input0 = uPred[:,0,:].unsqueeze(1).detach()
                input = torch.cat([input,  input0], dim=1)
            
            # Only do the first mini-batch
            break

    return u_out, betas, u_target

def plotContourGrid(t, xT, uPred, betas, uTarget):
    '''
    Creates grid of 4 different test cases, plots target, prediction, variance and error for each
    '''
    mpl.rcParams['font.family'] = ['serif'] # default is sans-serif
    rc('text', usetex=False)

    fig = plt.figure(figsize=(15, 13), dpi=150)
    outer = gridspec.GridSpec(2, 2, wspace=0.45, hspace=0.2) # Outer grid
    for i in range(4):
        # Inner grid
        inner = gridspec.GridSpecFromSubplotSpec(4, 1, 
            subplot_spec=outer[i], wspace=0, hspace=0.25)
        ax = []
        for j in range(4):
            ax0 = plt.Subplot(fig, inner[j])
            fig.add_subplot(ax0)
            ax.append(ax0)
        # Plot specific test case
        plotPred(fig, ax, t, xT, uPred[i], betas, uTarget[i])

    file_dir = '.'
    # If directory does not exist create it
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)
    file_name = file_dir+"/burger_BAR_pred"
    plt.savefig(file_name+".png", bbox_inches='tight')
    plt.savefig(file_name+".pdf", bbox_inches='tight')

    plt.show()

def plotPred(fig, ax, t, xT, uPred, betas, uTarget):
    '''
    Plots specific test case
    Args:
        fig: matplotlib figure
        ax (list): list of four subplot axis
        t (np.array): [n] array to time values for x axis
        xT (np.array): [m] array of spacial coordinates for y axis
        uPred (np.array): [n x m] model predictions
        uTarget (np.array): [n x m] target field
    '''
    # Start with the target up top
    cmap = "inferno"
    c0 = ax[0].imshow(uTarget.T, interpolation='nearest', cmap=cmap, origin='lower', aspect='auto', extent=[t[0],t[-1],xT[0],xT[-1]])
    c_max = np.max(uTarget.T)
    c_min = np.min(uTarget.T)
    c0.set_clim(vmin=c_min, vmax=c_max)

    # Plot the mean
    uPred_mean = np.mean(uPred, axis=0)
    c0 = ax[1].imshow(uPred_mean.T, interpolation='nearest', cmap=cmap, origin='lower', aspect='auto', extent=[t[0],t[-1],xT[0],xT[-1]])
    c0.set_clim(vmin=c_min, vmax=c_max)

    p0 = ax[0].get_position().get_points().flatten()
    p1 = ax[1].get_position().get_points().flatten()
    ax_cbar = fig.add_axes([p1[2]+0.015, p1[1], 0.020, p0[3]-p1[1]])
    ticks = np.linspace(0, 1, 5)
    tickLabels = np.linspace(c_min, c_max, 5)
    tickLabels = ["{:02.2f}".format(t0) for t0 in tickLabels]
    cbar = mpl.colorbar.ColorbarBase(ax_cbar, cmap=plt.get_cmap(cmap), orientation='vertical', ticks=ticks)
    cbar.set_ticklabels(tickLabels)

    # Variance
    betas = np.expand_dims(betas, axis=1).repeat(uPred.shape[1], axis=1) # Expand noise parameter
    betas = np.expand_dims(betas, axis=2).repeat(uPred.shape[2], axis=2) # Expand noise parameter
    uPred_var = np.mean(1./betas + uPred*uPred, axis=0) - uPred_mean*uPred_mean

    c0 = ax[2].imshow(uPred_var.T, interpolation='nearest', cmap=cmap, origin='lower', aspect='auto', extent=[t[0],t[-1],xT[0],xT[-1]])
    c_max = np.max(uPred_var)
    c0.set_clim(vmin=0, vmax=c_max)

    p0 = ax[2].get_position().get_points().flatten()
    ax_cbar = fig.add_axes([p0[2]+0.015, p0[1], 0.020, p0[3]-p0[1]])
    ticks = np.linspace(0, 1, 5)
    tickLabels = np.linspace(0, c_max, 5)
    tickLabels = ["{:02.2f}".format(t0) for t0 in tickLabels]
    cbar = mpl.colorbar.ColorbarBase(ax_cbar, cmap=plt.get_cmap(cmap), orientation='vertical', ticks=ticks)
    cbar.set_ticklabels(tickLabels)

    # Mean Error
    cmap = "viridis"
    c0 = ax[3].imshow(np.abs(uPred_mean.T - uTarget.T), interpolation='nearest', cmap=cmap, origin='lower', aspect='auto', extent=[t[0],t[-1],xT[0],xT[-1]])
    p0 = ax[3].get_position().get_points().flatten()
    ax_cbar = fig.add_axes([p0[2]+0.015, p0[1], 0.020, p0[3]-p0[1]])
    ticks = np.linspace(0, 1, 5)
    tickLabels = np.linspace(c0.norm.vmin, c0.norm.vmax, 5)
    tickLabels = ["{:.2e}".format(t0) for t0 in tickLabels]
    cbar = mpl.colorbar.ColorbarBase(ax_cbar, cmap=plt.get_cmap(cmap), orientation='vertical', ticks=ticks)
    cbar.set_ticklabels(tickLabels)

    ax[0].set_ylabel('x', fontsize=14)
    ax[1].set_ylabel('x', fontsize=14)
    ax[2].set_ylabel('x', fontsize=14)
    ax[3].set_ylabel('x', fontsize=14)
    ax[3].set_xlabel('t', fontsize=14)
    
    # Remove some tick labels to help declutter plot
    ax[0].set_xticklabels([])
    ax[1].set_xticklabels([])
    ax[2].set_xticklabels([])
    for ax0 in ax:
        ax0.set_yticks([0,0.5,1])

if __name__ == '__main__':

    # Parse arguements
    args = Parser().parse(dirs=False)
    use_cuda = "cpu"
    if(torch.cuda.is_available()):
        use_cuda = "cuda"
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Torch device:{}".format(args.device))
    
    # Domain settings, matches solver settings
    x0 = 0
    x1 = 1.0
    args.dx = (x1 - x0)/args.nel

    # Create training loader
    burgerLoader = BurgerLoader(dt=args.dt)
    # Create training loader
    test_cases = np.arange(0,5,1).astype(int)
    testing_loader = burgerLoader.createTestingLoader('../solver/fenics_data_dt0.001_T2.0', test_cases, batch_size=5)

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
    # Stochastic weighted averages
    swag_nn = SwagNN(args, bayes_nn, full_cov=True, max_models=args.swag_max)
    # Load network
    swag_nn.loadModel(200, file_dir="./networks")

    with torch.no_grad():
        uPred, betas, uTarget = testSample(args, swag_nn, testing_loader, tstep=400, n_samples=30)

    tTest = np.arange(0, 400*args.dt+1e-8, args.dt)
    xTest = np.linspace(x0, x1, args.nel+1)

    plotContourGrid(tTest, xTest, uPred.cpu().numpy(), betas.cpu().numpy(), uTarget.cpu().numpy())

