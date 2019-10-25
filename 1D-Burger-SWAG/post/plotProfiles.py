'''
This script plots predictive profiles of a given test case
at several different times for the FEM simulations, FDM simulations
and BAR-DenseED. NOTE: This graphic requires simulation data from
several simulations at different time-step sizes! The produced
graphic is seen in Figure 14 and 15 of the paper.
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

def readSimulatorData(data_dir, cases, t_start=0, t_range=[0,2], nn_dt=0.005, dt=0.001):
    '''
    Reads in simulator (FEM or FDM) data
    '''
    data=[]
    for i, val in enumerate(cases):
        file_name = data_dir+"/u{:d}.npy".format(val)
        print("Reading file: {}".format(file_name))
        u = np.load(file_name)
        # Remove last element due to periodic conditions between [0,1]
        uNp = u[::int(nn_dt/dt), :-1]
        data.append(uNp)

    return np.stack(data, axis=0)

def testSample(args, swag_nn, test_loader, tstep=100, n_samples=10):
    '''
    Tests smaples of the model using SWAG of the first test case in the testing loader
    Args:
        model (PyTorch model): DenseED model to be tested
        device (PtTorch device): device model is on
        test_loader (dataloader): dataloader with test cases with no shuffle (use createTestingLoader)
        tstep (int): number of timesteps to predict for
        n_samples (int): number of model samples to draw
    Returns:
        uPred (torch.Tensor): [mb x n_samp x t x n] predicted quantities
        betas (np.array): [mb x n_samp] predicted additive output noise
        u_target (torch.Tensor): [mb x t x n] target/ simulated values
    '''
    
    mb_size = int(len(test_loader.dataset)/len(test_loader))
    u_out = torch.zeros(mb_size, n_samples, tstep+1, args.nel)
    betas = torch.zeros(n_samples)
    
    for i in range(n_samples):
        print('Executing model sample {:d}'.format(i))
        model = swag_nn.sample(diagCov=False)
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

def plotProfiles(t, xT, uPred, betas, uSimData, uSimTitles, uSimLineType, dt=0.005, case = 0):
    '''
    Plots specific test case
    Args:
        t (np.array): [n] array to time values for x axis
        xT (np.array): [m] array of spacial coordinates for y axis
        uPred (np.array): [n x m] model predictions
        uSimData (list): list of np.arrays containing various predictions
            using different numerical integrators
        uSimTitles (list): list of titles associated with simulator data
        dt (float): time-step size of model
        case (int): which test case to plot, default is 0
    '''
    # Create figure
    mpl.rcParams['font.family'] = ['serif'] # default is sans-serif
    rc('text', usetex=False)

    fig = plt.figure(figsize=(10, 4), dpi=150)
    ax = []
    ax.append(plt.subplot2grid((12, 20), (0, 0), colspan=3, rowspan=6))
    ax.append(plt.subplot2grid((12, 20), (0, 4), colspan=3, rowspan=6))
    ax.append(plt.subplot2grid((12, 20), (0, 8), colspan=3, rowspan=6))
    ax.append(plt.subplot2grid((12, 20), (0, 12), colspan=3, rowspan=6))
    ax.append(plt.subplot2grid((12, 12), (8, 0), colspan=11, rowspan=4))

    # Start with simulator data
    pTimes = [0.1, 0.25, 0.75, 1.5]
    colors = cm.gist_rainbow(np.linspace(0, 1, len(uSimData)+1))
    for i, uTarget in enumerate(uSimData):
        for j, t0 in enumerate(pTimes):
            u0 = np.concatenate([uTarget[case, int(t0/dt)], uTarget[case, int(t0/dt), -1:]])
            ax[j].plot(xT, u0, uSimLineType[i], c=colors[i], label=uSimTitles[i])
            if(case == 0):
                ax[j].set_ylim([-1.25, 2.0])
            elif(case ==  1):
                ax[j].set_ylim([-1.5, 1.25])
            else:
                ax[j].set_ylim([-1.5, 1.0])
            ax[j].set_title('t={:.02f}'.format(t0))
            ax[j].set_xticks([0, 0.5, 1.0])
            ax[j].set_xlabel('x')

    # Surrogate profiles
    uPred_mean = np.mean(uPred[case], axis=0)
    # Variance
    betas = np.expand_dims(betas, axis=1).repeat(uPred[case].shape[1], axis=1) # Expand noise parameter
    betas = np.expand_dims(betas, axis=2).repeat(uPred[case].shape[2], axis=2) # Expand noise parameter
    uPred_std = np.sqrt(np.abs(np.mean(1./betas + uPred[case]*uPred[case], axis=0) - uPred_mean*uPred_mean))
    for i, t0 in enumerate(pTimes):
        u0 = np.concatenate([uPred_mean[int(t0/dt)], uPred_mean[int(t0/dt), -1:]])
        ax[i].plot(xT, u0, c=colors[-1], label='NN Mean')

        u0_std = np.concatenate([uPred_std[int(t0/dt)], uPred_std[int(t0/dt), -1:]])
        ax[i].fill_between(xT, u0+2*u0_std, u0-2*u0_std, facecolor='b', alpha=0.3, label=r'NN $2\sigma$')

    # Misc. unique labels
    ax[0].set_ylabel('u')
    ax[3].legend(bbox_to_anchor=(1.0, 1.05))

    # Plot the target solution contour
    cmap = "inferno"
    c0 = ax[4].imshow(uSimData[0][case].T, interpolation='nearest', cmap=cmap, origin='lower', aspect='auto', extent=[t[0],t[-1],xT[0],xT[-1]])
    c_max = np.max(uSimData[0][case].T)
    c_min = np.min(uSimData[0][case].T)
    c0.set_clim(vmin=c_min, vmax=c_max)
    ax[4].set_xlabel('t')
    ax[4].set_ylabel('x')

    # Color bar
    p0 = ax[4].get_position().get_points().flatten()
    ax_cbar = fig.add_axes([p0[2]+0.015, p0[1], 0.020, p0[3]-p0[1]])
    ticks = np.linspace(0, 1, 5)
    tickLabels = np.linspace(c_min, c_max, 5)
    tickLabels = ["{:02.2f}".format(t0) for t0 in tickLabels]
    cbar = mpl.colorbar.ColorbarBase(ax_cbar, cmap=plt.get_cmap(cmap), orientation='vertical', ticks=ticks)
    cbar.set_ticklabels(tickLabels)

    for i, t0 in enumerate(pTimes):
        ax[4].plot([t0, t0], [xT[0], xT[-1]], c='b')

    file_dir = '.'
    # If directory does not exist create it
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)
    file_name = file_dir+"/burger_BAR_profiles_{:d}".format(case)
    plt.savefig(file_name+".png", bbox_inches='tight')
    plt.savefig(file_name+".pdf", bbox_inches='tight')

    

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
    testing_loader = burgerLoader.createTestingLoader('../solver/fenics_data_dt0.001_T2.0', test_cases, dt=0.001, batch_size=5)

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
        uPred, betas, uTarget = testSample(args, swag_nn, testing_loader, tstep=400, n_samples=30)

    tTest = np.arange(0, 400*args.dt+1e-8, args.dt)
    xTest = np.linspace(x0, x1, args.nel+1)

    # With the neural network simulated, now load numerical simulators
    # Finite element
    uFEM1 = readSimulatorData('../solver/fenics_data_dt0.0005_T2.0', test_cases, nn_dt=0.005, dt=0.0005)
    uFEM2 = readSimulatorData('../solver/fenics_data_dt0.001_T2.0', test_cases, nn_dt=0.005, dt=0.001)
    uFEM3 = readSimulatorData('../solver/fenics_data_dt0.005_T2.0', test_cases, nn_dt=0.005, dt=0.005)
    # Finite difference
    uFDM1 = readSimulatorData('../solver/fd_data_dt0.0001_T2.0', test_cases, nn_dt=0.005, dt=0.0001)
    uFDM2 = readSimulatorData('../solver/fd_data_dt0.0005_T2.0', test_cases, nn_dt=0.005, dt=0.0005)

    # Make lists
    uSimData = [uFEM1, uFEM2, uFEM3, uFDM1, uFDM2]
    uSimTitle = [r'FEM $\Delta t = 0.0005$', r'FEM $\Delta t = 0.001$', r'FEM $\Delta t = 0.005$',\
        r'FDM $\Delta t = 0.0001$', r'FDM $\Delta t = 0.0005$']
    uSimLineType = ['-', '-', '-', '--', '--']

    # uSimData = [uFEM2]
    # uSimTitle = [r'FEM $\Delta t = 0.001$']
    # uSimLineType = ['-']

    # case = 0  for figure 14, case = 1 for figure 15
    plt.close("all")
    plotProfiles(tTest, xTest, uPred.cpu().numpy(), betas.cpu().numpy(), uSimData, uSimTitle, uSimLineType, case = 0)
    plotProfiles(tTest, xTest, uPred.cpu().numpy(), betas.cpu().numpy(), uSimData, uSimTitle, uSimLineType, case = 1)
    plt.show()