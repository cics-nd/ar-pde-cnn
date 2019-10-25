'''
This script plots the four test predictions of the predictive
expectation and variance of BAR-DenseED seen in Figure 23 and 24 of the paper.
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
    Tests samples of the model using SWAG of the first test min-batch in the testing loader
    Args:
        args (argparse): object with programs arguements
        model (PyTorch model): DenseED model to be tested
        test_loader (dataloader): dataloader with test cases with no shuffle (use createTestingLoader)
        tstep (int): number of timesteps to predict for
        n_samples (int): number of model samples to draw
    Returns:
        uPred (torch.Tensor): [mb x n_samp x t x n] predicted quantities
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
                uPred = model(input[:,:,:])

                u_out[:,i,t_idx+1,:,:, :] = uPred

                input = input[:,-2*int(args.nic-1):,:].detach()
                input0 = uPred.detach()
                input = torch.cat([input,  input0], dim=1)

            break
            
    return u_out, u_target, betas

def plotPred(args, case, uPred, uTarget, beta, target_steps, pred_steps, epoch=0):
    '''
    Plots several timesteps of the 2D Burger system
    '''
    # Create figure
    mpl.rcParams['font.family'] = ['serif'] # default is sans-serif
    rc('text', usetex=False)
    cmap = "plasma"
    cmap_error = "viridis"

    target = uTarget[target_steps]
    prediction = uPred[:,pred_steps]
    mean_pred = np.mean(prediction, axis=0)
    mean2_pred = np.mean(prediction**2, axis=0)
    mean_beta = np.mean(beta)
    error = np.abs(mean_pred - target)

    print(f'epoch {epoch}, plot prediction {case}')
    fig, ax = plt.subplots(8, len(pred_steps), figsize=(len(pred_steps)*3, 15))
    fig.subplots_adjust(wspace=0.5)

    for i in range(len(pred_steps)):
        for j in range(2):

            c_max = np.max(np.array([target[i,j], mean_pred[i,j]]))
            c_min = np.min(np.array([target[i,j], mean_pred[i,j]]))
            ax[4*j, i].imshow(target[i,j], interpolation='nearest', cmap=cmap, origin='lower', aspect='auto', extent=[0,1,0,1], vmin=c_min, vmax=c_max)
            ax[4*j+1, i].imshow(mean_pred[i,j], interpolation='nearest', cmap=cmap, origin='lower', aspect='auto', extent=[0,1,0,1], vmin=c_min, vmax=c_max)

            var = 1./mean_beta + mean2_pred[i,j] - mean_pred[i,j]**2
            # Hard code zero error for initial state
            if(pred_steps[i] == 0): #
                var[:,:] = 0
                error[i,j] = 0

            ax[4*j+2, i].imshow(var, interpolation='nearest', cmap=cmap, origin='lower', aspect='auto', extent=[0,1,0,1])
            c_max_var = np.max(var)
            ax[4*j+3, i].imshow(error[i,j], interpolation='nearest', cmap=cmap_error, origin='lower', aspect='auto', extent=[0,1,0,1])
            c_max_error = np.max(error[i,j])
            c_min_error = np.min(error[i,j])

            p0 =ax[4*j, i].get_position().get_points().flatten()
            p1 = ax[4*j+1, i].get_position().get_points().flatten()
            ax_cbar = fig.add_axes([p1[2]+0.0075, p1[1], 0.005, p0[3]-p1[1]])
            ticks = np.linspace(0, 1, 5)
            tickLabels = np.linspace(c_min, c_max, 5)
            tickLabels = ["{:02.2f}".format(t0) for t0 in tickLabels]
            cbar = mpl.colorbar.ColorbarBase(ax_cbar, cmap=plt.get_cmap(cmap), orientation='vertical', ticks=ticks)
            cbar.set_ticklabels(tickLabels)

            p0 =ax[4*j+2, i].get_position().get_points().flatten()
            ax_cbar = fig.add_axes([p0[2]+0.0075, p0[1], 0.005, p0[3]-p0[1]])
            ticks = np.linspace(0, 1, 5)
            tickLabels = np.linspace(0, c_max_var, 5)
            tickLabels = ["{:02.2f}".format(t0) for t0 in tickLabels]
            cbar = mpl.colorbar.ColorbarBase(ax_cbar, cmap=plt.get_cmap(cmap), orientation='vertical', ticks=ticks)
            cbar.set_ticklabels(tickLabels)

            p0 =ax[4*j+3, i].get_position().get_points().flatten()
            ax_cbar = fig.add_axes([p0[2]+0.0075, p0[1], 0.005, p0[3]-p0[1]])
            ticks = np.linspace(0, 1, 5)
            tickLabels = np.linspace(c_min_error, c_max_error, 5)
            tickLabels = ["{:02.2f}".format(t0) for t0 in tickLabels]
            cbar = mpl.colorbar.ColorbarBase(ax_cbar, cmap=plt.get_cmap(cmap_error), orientation='vertical', ticks=ticks)
            cbar.set_ticklabels(tickLabels)

            
            for ax0 in ax[:-1,i]:
                ax0.set_xticklabels([])

            for ax0 in ax[:,i]:
                ax0.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
                ax0.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
                if(i > 0):
                    ax0.set_yticklabels([])
                else:
                    ax0.set_ylabel('y', fontsize=14)

        ax[0, i].set_title('t={:.02f}'.format(pred_steps[i] * args.dt), fontsize=14)
        ax[-1, i].set_xlabel('x', fontsize=14)

    file_dir = '.'
    # If director does not exist create it
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)
    file_name = file_dir+"/burger2D_BAR_pred{:d}".format(case)

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
    swag_nn = SwagNN(args, bayes_nn, full_cov=True, max_models=args.swag_max)
    # Load the model
    swag_nn.loadModel(200, file_dir='./networks')

    n_test = 150
    with torch.no_grad():
        uPred, uTarget, betas = testSample(args, swag_nn, testing_loader, tstep=n_test, n_samples=30)
    
    step = 20
    pred_steps = np.arange(0, n_test+1, step)
    target_steps = pred_steps//2
    
    plt.close("all")
    case = 0
    plotPred(args, test_cases[case], uPred[case].cpu().numpy(), uTarget[case].cpu().numpy(), betas.cpu().numpy(), target_steps, pred_steps)
    case = 1
    plotPred(args, test_cases[case], uPred[case].cpu().numpy(), uTarget[case].cpu().numpy(), betas.cpu().numpy(), target_steps, pred_steps)

    plt.show()