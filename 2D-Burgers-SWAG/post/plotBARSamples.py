'''
This script plots samples from the approximated posterior of 
BAR-DenseED for the 2D Burgers system. The produced graphic 
is seen in Figure 21 and 22 of the paper.
===
Distributed by: Notre Dame CICS (MIT Liscense)
- Associated publication:
url: http://www.sciencedirect.com/science/article/pii/S0021999119307612
doi: https://doi.org/10.1016/j.jcp.2019.109056
github: https://github.com/cics-nd/ar-pde-cnn
===
'''
import sys
sys.path.append('..')
from utils.utils import mkdirs

from args import Parser
from nn.denseEDcirc2d import DenseED
from nn.bayesNN import BayesNN
from nn.swag import SwagNN
from utils.utils import mkdirs, toNumpy, toTuple
from utils.burgerLoader2D import BurgerLoader

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib import cm

import torch
import numpy as np
import os, time

def testSample(args, swag_nn, test_loader, tstep=100, n_samples=10):
    '''
    Tests samples of the model using SWAG of the first test mini-batch in the testing loader
    Args:
        args (argparse): object with programs arguements
        model (PyTorch model): DenseED model to be tested
        test_loader (dataloader): dataloader with test cases with no shuffle (use createTestingLoader)
        tstep (int): number of timesteps to predict for
        n_samples (int): number of model samples to draw, default 10
    Returns:
        uPred (torch.Tensor): [mb x n_samp x t x n] predicted quantities
        u_target (torch.Tensor): [mb x t x n] target/ simulated values
    '''
    
    mb_size = int(len(test_loader.dataset)/len(test_loader))
    u_out = torch.zeros(mb_size, n_samples, tstep+1, 2, args.nel, args.nel)
    
    for i in range(n_samples):
        print('Executing model sample {:d}'.format(i))
        model = swag_nn.sample(diagCov=False)
        model.eval()

        for batch_idx, (input0, uTarget0) in enumerate(test_loader):
            # Expand input to match model in channels
            dims = torch.ones(len(input0.shape))
            dims[1] = args.nic
            input = input0.repeat(toTuple(toNumpy(dims).astype(int))).to(args.device)
            
            u_target = uTarget0
            u_out[:,i,0,:,:,:] = input0
            # Auto-regress
            for t_idx in range(tstep):
                uPred = model(input[:,-2*args.nic:,:])

                u_out[:,i,t_idx+1,:,:, :] = uPred

                input = input[:,-2*int(args.nic-1):,:].detach()
                input0 = uPred.detach()
                input = torch.cat([input,  input0], dim=1)
            
            # Only do the first mini-batch
            break

    return u_out, u_target


def plotSamples(args, case, uPred, uTarget, tstep=1):
    '''
    Plots samples of one timestep
    Args:
        args (argparse): object with programs arguements
        case (int): case number to ploat
        uPred (torch.Tensor): [mb x n_samp x t x n] predicted quantities
        u_target (torch.Tensor): [mb x t x n] target/ simulated values
        tstep (int): time-step to plot
    '''
    plt.close("all")
    # Create figure
    mpl.rcParams['font.family'] = ['serif'] # default is sans-serif
    rc('text', usetex=True)

    n_sample = uPred.shape[0] + 1
    nrow = int(np.sqrt(n_sample))
    ncol = 12*nrow + 3
    fig = plt.figure(figsize=(25, 10), dpi=150)
    axu = []
    axv = []
    for i in range(nrow):
        for j in range(nrow):
            axu.append(plt.subplot2grid((nrow, ncol), (i, 6*j), colspan=5))
            axv.append(plt.subplot2grid((nrow, ncol), (i, 6*j + 6*nrow + 3), colspan=5))

    def plotGrid(fig, ax, uTarget, uPred):
        cmap = "plasma" 
        # Target in top left
        c0 = ax[0].imshow(uTarget, interpolation='nearest', cmap=cmap, origin='lower', aspect='auto', extent=[0,1,0,1])
        c_max = np.max(uPred.T)
        c_min = np.min(uPred.T)
        c0.set_clim(vmin=c_min, vmax=c_max)

        # Prediction samples
        for i in range(1, len(ax)):
            c0 = ax[i].imshow(uPred[i-1], interpolation='nearest', cmap=cmap, origin='lower', aspect='auto', extent=[0,1,0,1])
            c0.set_clim(vmin=c_min, vmax=c_max)

        p0 = ax[nrow-1].get_position().get_points().flatten()
        p1 = ax[-1].get_position().get_points().flatten()
        ax_cbar = fig.add_axes([p1[2]+0.01, p1[1], 0.015, p0[3]-p1[1]])
        ticks = np.linspace(0, 1, 5)
        tickLabels = np.linspace(c_min, c_max, 5)
        tickLabels = ["{:02.2f}".format(t0) for t0 in tickLabels]
        cbar = mpl.colorbar.ColorbarBase(ax_cbar, cmap=plt.get_cmap(cmap), orientation='vertical', ticks=ticks)
        cbar.set_ticklabels(tickLabels)

        for ax0 in ax:
            ax0.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
            ax0.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        # Axis labels
        for i in range(len(ax)-nrow, len(ax)):
            ax[i].set_xlabel('y', fontsize=14)
        for i in range(nrow):
            ax[int(i*nrow)].set_ylabel('x', fontsize=14)

    plotGrid(fig, axu, uTarget[tstep, 0], uPred[:,2*tstep,0])
    plotGrid(fig, axv, uTarget[tstep, 1], uPred[:,2*tstep,1])
    plt.suptitle('T: {:03f}'.format(args.dt*2*tstep))

    file_dir = '.'
    # If directory does not exist create it
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)
    file_name = file_dir+"/burger2D_BAR_samples{:d}_t{:d}".format(case, tstep)

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
    
    # Domain settings, matches solver settings
    x0 = 0
    x1 = 1.0
    args.dx = (x1 - x0)/args.nel

    # Create training loader
    burgerLoader = BurgerLoader(dt=args.dt)
    # Create training loader
    test_cases = np.array([0, 1]).astype(int)
    testing_loader = burgerLoader.createTestingLoader('../solver/fenics_data', test_cases, simdt=0.005, batch_size=2)

    # Create DenseED model
    denseED = DenseED(in_channels=2*args.nic, out_channels=2*args.noc,
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
    # Load the model
    swag_nn.loadModel(200, file_dir='./networks')

    n_test = 101 # Time-steps to test for 
    with torch.no_grad():
        uPred, uTarget = testSample(args, swag_nn, testing_loader, tstep=n_test, n_samples=8)
        # Plot the samples
        for t in [10, 50]:
            bidx = 0
            plotSamples(args, bidx, uPred[bidx].detach().numpy(), uTarget[bidx].cpu().numpy(), tstep=t)