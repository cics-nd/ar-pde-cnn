'''
This script plots the four test predictions of the auto-regressive DenseED
seen in Figure 10 of the paper.
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

def test(args, model, test_loader, tstep=100):
    '''
    Tests the base derterministic model of a single mini-batch
    Args:
        args (argparse): object with programs arguements
        model (PyTorch model): DenseED model to be tested
        test_loader (dataloader): dataloader with test cases (use createTestingLoader)
        tstep (int): number of timesteps to predict for
    Returns:
        uPred (torch.Tensor): [mb x t x n] predicted quantities
    '''
    model.eval()
    mb_size = int(len(test_loader.dataset)/len(test_loader))
    u_out = torch.zeros(mb_size, tstep+1, args.nel).to(args.device)

    for bidx, (input0, uTarget0) in enumerate(test_loader):
        input = input0.to(args.device)
        if(bidx == 0):
            u_out[:,0,:] = input[:,0]
            u_target = uTarget0
        # Auto-regress
        for t_idx in range(tstep):
            uPred = model(input[:,-args.nic:,:])
            if(bidx == 0):
                u_out[:,t_idx+1,:] = uPred[:,0]
            
            input = input[:,-(args.nic-1):,:].detach()
            input0 = uPred[:,0,:].unsqueeze(1).detach()
            input = torch.cat([input,  input0], dim=1)
        break

    return u_out, u_target

def plotContourGrid(t, xT, uPred, uTarget):
    '''
    Creates grid of 4 different test cases, plots target, prediction and error for each
    '''
    mpl.rcParams['font.family'] = ['serif'] # default is sans-serif
    rc('text', usetex=False)

    fig = plt.figure(figsize=(15, 9), dpi=150)
    outer = gridspec.GridSpec(2, 2, wspace=0.45, hspace=0.2) # Outer grid
    for i in range(4):
        # Inner grid
        inner = gridspec.GridSpecFromSubplotSpec(3, 1, 
            subplot_spec=outer[i], wspace=0, hspace=0.2)
        ax = []
        for j in range(3):
            ax0 = plt.Subplot(fig, inner[j])
            fig.add_subplot(ax0)
            ax.append(ax0)
        # Plot specific test case
        plotPred(fig, ax, t, xT, uPred[i], uTarget[i])

    file_dir = '.'
    # If directory does not exist create it
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)
    file_name = file_dir+"/burger_AR_pred"
    plt.savefig(file_name+".png", bbox_inches='tight')
    plt.savefig(file_name+".pdf", bbox_inches='tight')

    plt.show()


def plotPred(fig, ax, t, xT, uPred, uTarget, cmap = "inferno", err_cmap = "viridis"):
    '''
    Plots specific test case
    Args:
        fig: matplotlib figure
        ax (list): list of three subplot axis
        t (np.array): [n] array to time values for x axis
        xT (np.array): [m] array of spacial coordinates for y axis
        uPred (np.array): [n x m] model predictions
        uTarget (np.array): [n x m] target field
    '''
    c0 = ax[1].imshow(uPred.T, interpolation='nearest', cmap=cmap, origin='lower', aspect='auto', extent=[t[0],t[-1],xT[0],xT[-1]])
    c_max = np.max(uPred.T)
    c_min = np.min(uPred.T)
    c0.set_clim(vmin=c_min, vmax=c_max)

    c0 = ax[0].imshow(uTarget.T, interpolation='nearest', cmap=cmap, origin='lower', aspect='auto', extent=[t[0],t[-1],xT[0],xT[-1]])
    c0.set_clim(vmin=c_min, vmax=c_max)

    p0 = ax[0].get_position().get_points().flatten()
    p1 = ax[1].get_position().get_points().flatten()
    ax_cbar = fig.add_axes([p1[2]+0.015, p1[1], 0.020, p0[3]-p1[1]])
    ticks = np.linspace(0, 1, 5)
    tickLabels = np.linspace(c_min, c_max, 5)
    tickLabels = ["{:02.2f}".format(t0) for t0 in tickLabels]
    cbar = mpl.colorbar.ColorbarBase(ax_cbar, cmap=plt.get_cmap(cmap), orientation='vertical', ticks=ticks)
    cbar.set_ticklabels(tickLabels)

    c0 = ax[2].imshow(np.abs(uPred.T - uTarget.T), interpolation='nearest', cmap=err_cmap, origin='lower', aspect='auto', extent=[t[0],t[-1],xT[0],xT[-1]])
    p0 = ax[2].get_position().get_points().flatten()
    ax_cbar = fig.add_axes([p0[2]+0.015, p0[1], 0.020, p0[3]-p0[1]])
    ticks = np.linspace(0, 1, 5)
    tickLabels = np.linspace(c0.norm.vmin, c0.norm.vmax, 5)
    tickLabels = ["{:.2e}".format(t0) for t0 in tickLabels]
    cbar = mpl.colorbar.ColorbarBase(ax_cbar, cmap=plt.get_cmap(err_cmap), orientation='vertical', ticks=ticks)
    cbar.set_ticklabels(tickLabels)

    ax[0].set_xticklabels([])
    ax[1].set_xticklabels([])

    ax[0].set_ylabel('x', fontsize=14)
    ax[1].set_ylabel('x', fontsize=14)
    ax[2].set_ylabel('x', fontsize=14)
    ax[2].set_xlabel('t', fontsize=14)

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
    swag_nn = SwagNN(args, bayes_nn, full_cov=True, max_models=30)
    # Load network
    swag_nn.loadModel(100, file_dir="./networks")

    with torch.no_grad():
        uPred, uTarget = test(args, swag_nn.base, testing_loader, tstep=400)

    tTest = np.arange(0, 400*args.dt+1e-8, args.dt)
    xTest = np.linspace(x0, x1, args.nel+1)

    plotContourGrid(tTest, xTest, uPred.cpu().numpy(), uTarget.cpu().numpy())

