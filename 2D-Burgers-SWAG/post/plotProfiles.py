'''
This script plots predictive profiles of a given test case
at several different times for the FEM simulations and BAR-DenseED.
The produced graphic is seen in Figure 25 of the paper.
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
from args import Parser
from nn.denseEDcirc2d import DenseED
from nn.bayesNN import BayesNN
from nn.swag import SwagNN
from utils.burgerLoader2D import BurgerLoader
from utils.utils import toNumpy, toTuple

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib import cm

import torch
import numpy as np
import os, time


def readSimulatorData(data_dir, cases, t_range=[0,1], nndt=0.005, simdt=0.005, save_every=2):
    '''
    Reads in simulator (FEM or FDM) data
    '''
    sim_data =[]
    # Loop through test cases
    for i, val in enumerate(cases):
        case_dir = os.path.join(data_dir, "run{:d}".format(val))
        print("Reading test case: {}".format(case_dir))
        seq = []
        for j in range(t_range[0], int(t_range[-1]/simdt)+1, save_every):
            file_dir = os.path.join(case_dir, "u{:d}.npy".format(j))
            u0 = np.load(file_dir)
            # Remove the periodic nodes
            seq.append(u0[:,:,:])

        file_dir = os.path.join(case_dir, "u0.npy")
        uInit = np.load(file_dir)
        uTarget = np.stack(seq, axis=0)

        # Remove the periodic nodes and unsqueeze first dim
        sim_data.append(uTarget)
    
    return np.stack(sim_data, axis=0)

def testSample(args, swag_nn, test_loader, tstep=100, n_samples=10):
    '''
    Tests samples of the model using SWAG of the first test mini-batch in the testing loader
    Args:
        args (argparse): object with programs arguements
        model (PyTorch model): DenseED model to be tested
        test_loader (dataloader): dataloader with test cases with no shuffle (use createTestingLoader)
        tstep (int): number of timesteps to predict for
    Returns:
        uPred (torch.Tensor): [mb x n_samp x t x n] predicted quantities
        betas (np.array): [mb x n_samp] predicted additive output noise
        u_target (torch.Tensor): [mb x t x n] target/ simulated values
    '''
    mb_size = int(len(test_loader.dataset)/len(test_loader))
    u_out = torch.zeros(mb_size, n_samples, tstep+1, 2, args.nel, args.nel)
    betas = torch.zeros(n_samples)
    
    for i in range(n_samples):
        print('Executing model sample {:d}'.format(i))
        model = swag_nn.sample(diagCov=False)
        model.eval()
        betas[i] = model.model.log_beta.exp()
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
            
    return u_out, u_target, betas

def plotProfiles(tsteps, case, uPred, betas, uSimData, uSimTitles, uSimLineType, dt=0.01):
    '''
    Plots profiles of a specific test case
    Args:
        tsteps (np.array): [3] time-steps to plot profiles at
        case (int): which test case to plot
        uPred (np.array): [nsamp x m] model predictions
        betas (np.array): [nsamp] predicted additive output noise
        uSimData (list): list of np.arrays containing various predictions
            using different numerical integrators
        uSimTitles (list): list of titles associated with simulator data
        dt (float): time-step size of model
    '''
    # Create figure
    mpl.rcParams['font.family'] = ['serif'] # default is sans-serif
    rc('text', usetex=False)

    fig = plt.figure(figsize=(7*len(tsteps), 10))
    ax = [[] for i in range(2*len(tsteps))]
    for j in range(2*len(tsteps)):
        ax[j].append(plt.subplot2grid((12, 14*len(tsteps)), (0, 7*j), colspan=5, rowspan=3))
        ax[j].append(plt.subplot2grid((12, 14*len(tsteps)), (4, 7*j), colspan=5, rowspan=3))
        ax[j].append(plt.subplot2grid((12, 14*len(tsteps)), (8, 7*j), colspan=5, rowspan=3))
    ax = np.array(ax).T

    for i in range(len(tsteps)):
        # Plot the target solution contour
        cmap = "plasma"
        uTarget = uSimData[0]
        ax[2, 2*i].imshow(uTarget[case, tsteps[i], 0], interpolation='nearest', cmap=cmap, origin='lower', aspect='auto', extent=[0,1,0,1])
        ax[2, 2*i].set_xlabel('x', fontsize=14)
        ax[2, 2*i].set_ylabel('y', fontsize=14)
        ax[2, 2*i].plot([0, 1], [0.5, 0.5], c='b', linewidth=1.0)
        ax[2, 2*i].plot([0.5, 0.5], [0, 1], c='b', linewidth=1.0)

        ax[2, 2*i + 1].imshow(uTarget[case, tsteps[i], 1], interpolation='nearest', cmap=cmap, origin='lower', aspect='auto', extent=[0,1,0,1])
        ax[2, 2*i + 1].set_xlabel('x', fontsize=14)
        ax[2, 2*i + 1].set_ylabel('y', fontsize=14)
        ax[2, 2*i + 1].plot([0, 1], [0.5, 0.5], c='b', linewidth=1.0)
        ax[2, 2*i + 1].plot([0.5, 0.5], [0, 1], c='b', linewidth=1.0)

        ax[2, 2*i].set_xticks([0, 0.5, 1.0])
        ax[2, 2*i + 1].set_xticks([0, 0.5, 1.0])
        ax[2, 2*i].set_yticks([0, 0.5, 1.0])
        ax[2, 2*i + 1].set_yticks([0, 0.5, 1.0])

    # Start with simulator data
    colors = cm.gist_rainbow(np.linspace(0, 0.75, len(uSimData)+1))
    xT = np.linspace(0, 1, 64)
    for i, uTarget in enumerate(uSimData):
        for j, t0 in enumerate(tsteps):
            hidx = uTarget.shape[-1]//2
            # x-velocity
            u0 = uTarget[case, t0, 0, hidx, :]
            ax[0, 2*j].plot(xT, u0, uSimLineType[i], c=colors[i], label=uSimTitles[i])

            u0 = uTarget[case, t0, 0, :, hidx]
            ax[1, 2*j].plot(xT, u0, uSimLineType[i], c=colors[i], label=uSimTitles[i])

            # y-velocity
            v0 = uTarget[case, t0, 1, hidx, :]
            ax[0, 2*j + 1].plot(xT, v0, uSimLineType[i], c=colors[i], label=uSimTitles[i])

            v0 = uTarget[case, t0, 1, :, hidx]
            ax[1, 2*j + 1].plot(xT, v0, uSimLineType[i], c=colors[i], label=uSimTitles[i])

            if(j == 0 and i == 0):
                c_maxu = np.max([uTarget[case, t0, 0, hidx, :], uTarget[case, t0, 0, :, hidx]])
                c_minu = np.min([uTarget[case, t0, 0, hidx, :], uTarget[case, t0, 0, :, hidx]])
                c_maxv = np.max([uTarget[case, t0, 1, hidx, :], uTarget[case, t0, 1, :, hidx]])
                c_minv = np.min([uTarget[case, t0, 1, hidx, :], uTarget[case, t0, 1, :, hidx]])

            # Yikes
            ax[0, 2*j].set_xticks([0, 0.5, 1.0])
            ax[0, 2*j].set_xlabel('x', fontsize=14)
            ax[0, 2*j].set_ylabel('u', fontsize=14)
            ax[0, 2*j].set_ylim([c_minu - 0.25, c_maxu + 0.25])

            ax[1, 2*j].set_xticks([0, 0.5, 1.0])
            ax[1, 2*j].set_xlabel('y', fontsize=14)
            ax[1, 2*j].set_ylabel('u', fontsize=14)
            ax[1, 2*j].set_ylim([c_minu - 0.25, c_maxu + 0.25])

            ax[0, 2*j + 1].set_xticks([0, 0.5, 1.0])
            ax[0, 2*j + 1].set_xlabel('x', fontsize=14)
            ax[0, 2*j + 1].set_ylabel('v', fontsize=14)
            ax[0, 2*j + 1].set_ylim([c_minv - 0.25, c_maxv + 0.25])

            ax[1, 2*j + 1].set_xticks([0, 0.5, 1.0])
            ax[1, 2*j + 1].set_xlabel('y', fontsize=14)
            ax[1, 2*j + 1].set_ylabel('v', fontsize=14)
            ax[1, 2*j + 1].set_ylim([c_minv - 0.25, c_maxv + 0.25])

            p0 = ax[0, 2*j].get_position().get_points().flatten()
            p1 = ax[0, 2*j + 1].get_position().get_points().flatten()
            xloc = (p0[2]+p1[0])/2.
            yloc = p0[3] + 0.03
            plt.figtext(xloc, yloc, r'$t = {:.02f}$'.format(t0*dt), ha='center', family='serif', fontsize=14, usetex=True)

    # Surrogate profiles
    uPred_mean = np.mean(uPred[case], axis=0)
    # Variance
    betas = np.expand_dims(betas, axis=1).repeat(uPred[case].shape[1], axis=1) # Expand noise parameter
    betas = np.expand_dims(betas, axis=2).repeat(uPred[case].shape[2], axis=2) # Expand noise parameter
    betas = np.expand_dims(betas, axis=3).repeat(uPred[case].shape[3], axis=3) # Expand noise parameter
    betas = np.expand_dims(betas, axis=4).repeat(uPred[case].shape[4], axis=4) # Expand noise parameter
    uPred_std = np.sqrt(np.abs(np.mean(1./betas + uPred[case]*uPred[case], axis=0) - uPred_mean*uPred_mean))

    for i, t0 in enumerate(tsteps):
        # x-velocity
        u0 = uPred_mean[t0, 0, hidx, :]
        ax[0, 2*i].plot(xT, u0, c=colors[-1])
        u0_std = uPred_std[t0, 0, hidx, :]
        ax[0, 2*i].fill_between(xT, u0+2*u0_std, u0-2*u0_std, facecolor='b', alpha=0.3)

        u0 = uPred_mean[t0, 0, :, hidx]
        ax[1, 2*i].plot(xT, u0, c=colors[-1])
        u0_std = uPred_std[t0, 0, :, hidx]
        ax[1, 2*i].fill_between(xT, u0+2*u0_std, u0-2*u0_std, facecolor='b', alpha=0.3)

        # y-velocity
        u0 = uPred_mean[t0, 1, hidx, :]
        ax[0, 2*i + 1].plot(xT, u0, c=colors[-1], label='NN Mean')
        u0_std = uPred_std[t0, 1, hidx, :]
        ax[0, 2*i + 1].fill_between(xT, u0+2*u0_std, u0-2*u0_std, facecolor='b', alpha=0.3, label=r'NN $2\sigma$')

        u0 = uPred_mean[t0, 1, :, hidx]
        ax[1, 2*i + 1].plot(xT, u0, c=colors[-1])
        u0_std = uPred_std[t0, 1, :, hidx]
        ax[1, 2*i + 1].fill_between(xT, u0+2*u0_std, u0-2*u0_std, facecolor='b', alpha=0.3)

    # Legend in top right on
    ax[0,-1].legend(loc='lower right')

    file_dir = '.'
    # If director does not exist create it
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)
    file_name = file_dir+"/burger2D_BAR_profiles_{:d}".format(case)
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
    test_cases = np.array([2]).astype(int)
    testing_loader = burgerLoader.createTestingLoader('../solver/fenics_data', test_cases, simdt=0.005, batch_size=1)

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
    swag_nn = SwagNN(args, bayes_nn, full_cov=True, max_models=args.swag_max)
    # Load the model
    swag_nn.loadModel(200, file_dir='./networks')

    n_test = 200
    with torch.no_grad():
        uPred, uTarget, betas = testSample(args, swag_nn, testing_loader, tstep=n_test, n_samples=30)
    

    # With the neural network simulated, now load numerical simulators
    uFEM1 = readSimulatorData('../solver/fenics_data', test_cases, nndt=0.005)
    # Make lists
    uSimData = [uFEM1]
    uSimTitle = ['FEM']
    uSimLineType = ['-']

    tSteps = [20, 40, 60]
    plt.close("all")
    plotProfiles(tSteps, 0, uPred[:,:,::2].cpu().numpy(), betas.cpu().numpy(), uSimData, uSimTitle, uSimLineType)
    plt.show()