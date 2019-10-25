'''
This script plots samples from the approximated posterior of 
BAR-DenseED for the 1D Burgers system. The produced graphic 
is seen in Figure 12 of the paper.
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
    Tests samples of the model using SWAG of the first test case in the testing loader
    Args:
        model (PyTorch model): DenseED model to be tested
        device (PtTorch device): device model is on
        test_loader (dataloader): dataloader with test cases with no shuffle (use createTestingLoader)
        tstep (int): number of timesteps to predict for
        n_samples (int): number of model samples to draw, default 10
    Returns:
        uPred (torch.Tensor): [mb x n_samp x t x n] predicted quantities
        u_target (torch.Tensor): [mb x t x n] target/ simulated values
    '''
    
    mb_size = int(len(test_loader.dataset)/len(test_loader))
    u_out = torch.zeros(mb_size, n_samples, tstep+1, args.nel)
    
    for i in range(n_samples):
        print('Executing model sample {:d}'.format(i))
        model = swag_nn.sample(diagCov=False)
        model.eval()

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

    return u_out, u_target

def plotSamples(t, xT, uPred, uTarget, epoch=0):
    '''
    Plots prediction contour
    '''
    plt.close("all")
    # Create figure
    mpl.rcParams['font.family'] = ['serif'] # default is sans-serif
    # rc('text', usetex=True)

    n_sample = uPred.shape[0] + 1
    nrow = int(np.sqrt(n_sample))
    ncol = 6*nrow + 1
    fig = plt.figure(figsize=(20, 7), dpi=150)
    ax = []
    for i in range(nrow):
        for j in range(nrow):
            ax.append(plt.subplot2grid((nrow, ncol), (i, 6*j), colspan=5))

    cmap = "inferno" 
    # Target in top left
    uTarget = uTarget[:uPred.shape[1]]
    c0 = ax[0].imshow(uTarget.T, interpolation='nearest', cmap=cmap, origin='lower', aspect='auto', extent=[t[0],t[-1],xT[0],xT[-1]])
    c_max = np.max(uPred.T)
    c_min = np.min(uPred.T)
    c0.set_clim(vmin=c_min, vmax=c_max)

    # Prediction samples
    for i in range(1, len(ax)):
        c0 = ax[i].imshow(uPred[i-1].T, interpolation='nearest', cmap=cmap, origin='lower', aspect='auto', extent=[t[0],t[-1],xT[0],xT[-1]])
        c0.set_clim(vmin=c_min, vmax=c_max)

    p0 = ax[nrow-1].get_position().get_points().flatten()
    p1 = ax[-1].get_position().get_points().flatten()
    ax_cbar = fig.add_axes([p1[2]+0.01, p1[1], 0.020, p0[3]-p1[1]])
    ticks = np.linspace(0, 1, 5)
    tickLabels = np.linspace(c_min, c_max, 5)
    tickLabels = ["{:02.2f}".format(t0) for t0 in tickLabels]
    cbar = mpl.colorbar.ColorbarBase(ax_cbar, cmap=plt.get_cmap(cmap), orientation='vertical', ticks=ticks)
    cbar.set_ticklabels(tickLabels)

    # Axis labels
    for i in range(len(ax)-nrow, len(ax)):
        ax[i].set_xlabel('t')
    for i in range(nrow):
        ax[int(i*nrow)].set_ylabel('x')

    file_dir = '.'
    # If directory does not exist create it
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)
    file_name = file_dir+"/burger_BAR_Samples"
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
    test_cases = np.arange(5,10,1).astype(int)
    testing_loader = burgerLoader.createTestingLoader('../solver/fenics_data_dt0.001_T2.0', test_cases, batch_size=5)

    # Create DenseED model
    denseED = DenseED(in_channels=args.nic, out_channels=1,
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
    swag_nn.loadModel(200, file_dir="./networks")

    with torch.no_grad():
        uPred, uTarget = testSample(args, swag_nn, testing_loader, tstep=400)

    tTest = np.arange(0, 400*args.dt+1e-8, args.dt)
    xTest = np.linspace(x0, x1, args.nel+1)

    plotSamples(tTest, xTest, uPred[0].cpu().numpy(), uTarget[0].cpu().numpy())

