'''
This script plots samples from the approximated 
posterior of BAR-DenseED. The produced graphic 
is seen in Figure 8 of the paper.
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
import torch.nn.functional as F

import matplotlib as mpl
import numpy as np
import os
import scipy.integrate as integrate
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.lines import Line2D
from matplotlib import rc

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
        model = swag_nn.sample(diagCov=True)
        model.eval()

        for bidx, (input0, uTarget0) in enumerate(test_loader):
            input = input0.to(args.device)
            u_out[bidx*mb_size:(bidx+1)*mb_size, i, 0, :] = input[:,0]
            if(i == 0):
                u_target[bidx*mb_size:(bidx+1)*mb_size] = uTarget0[:,:tstep+1]
            # Auto-regress
            for tidx in range(tstep):
                uPred = model(input[:,-2:,:])
                u_out[bidx*mb_size:(bidx+1)*mb_size, i, tidx+1] = uPred[:,0].detach().cpu()
                
                input = input[:,-4:,:].detach()
                input0 = uPred[:,0,:].unsqueeze(1).detach()
                input = torch.cat([input,  input0], dim=1)

    return u_out, u_target

def plotSamples(args, t, xT, uPred, uTarget, case=0):
    '''
    Plots prediction contour
    '''
    plt.close("all")
    # Create figure
    mpl.rcParams['font.family'] = ['serif'] # default is sans-serif
    # rc('text', usetex=True)

    n_sample = uPred.shape[1] + 1
    nrow = int(np.sqrt(n_sample))
    fig = plt.figure(figsize=(15, 10), dpi=150)
    ax = []
    for i in range(nrow):
        for j in range(nrow):
            ax.append(plt.subplot2grid((nrow, nrow), (i, j)))

    cmap = "rainbow" 
    #Target
    c0 = ax[0].imshow(uTarget[case,:,:t.shape[0]].T, interpolation='nearest', cmap=cmap, origin='lower', aspect='auto', extent=[t[0],t[-1],xT[0],xT[-1]])
    c0.set_clim(vmin=-3, vmax=3)

    # Prediction samples
    for i in range(1, len(ax)):
        c0 = ax[i].imshow(uPred[case, i-1].T, interpolation='nearest', cmap=cmap, origin='lower', aspect='auto', extent=[t[0],t[-1],xT[0],xT[-1]])
        c0.set_clim(vmin=-3, vmax=3)

    p0 = ax[nrow-1].get_position().get_points().flatten()
    p1 = ax[-1].get_position().get_points().flatten()
    ax_cbar = fig.add_axes([p1[2]+0.02, p1[1], 0.020, p0[3]-p1[1]])
    ticks = np.linspace(0, 1, 5)
    tickLabels = np.linspace(-3, 3, 5)
    tickLabels = ["{:02.2f}".format(t0) for t0 in tickLabels]
    cbar = mpl.colorbar.ColorbarBase(ax_cbar, cmap=plt.get_cmap(cmap), orientation='vertical', ticks=ticks)
    cbar.set_ticklabels(tickLabels)

    # Axis labels
    for i in range(len(ax)-nrow, len(ax)):
        ax[i].set_xlabel('t', fontsize=14)
    for i in range(nrow):
        ax[int(i*nrow)].set_ylabel('x', fontsize=14)

    file_dir = '.'
    # If directory does not exist create it
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)

    file_name = file_dir+"/ks_BAR_Samples"
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
    
    # Domain settings
    nel = 96
    x0 = 0
    x1 =  22*np.pi
    deltaX = (x1 - x0)/nel

    # Create testing loaders
    ksLoader = KSLoader()
    test_cases = np.arange(96,100+1e-8,1).astype(int)
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
    swag_nn.loadModel(200, file_dir='networks')

    with torch.no_grad():
        uPred, uTarget = testSample(args, swag_nn, testing_loader, tstep=1000, n_samples=8)
    
    # Construct domain for plotting
    tTest = np.linspace(0, (1000)*args.dt, 1001)
    xTest = np.linspace(x0, x1, args.nel)

    plotSamples(args, tTest, xTest, uPred.detach().numpy(), uTarget.detach().numpy(), case=0)