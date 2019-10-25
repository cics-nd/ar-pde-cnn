'''
This script plots the average mean squared error (MSE) and 
energy squared error (ESE) as a function of time for a test 
set of 200 cases using AR-DenseED and predictive expectation 
of BAR-DenseED. The produced graphic is seen in Figure 20 of 
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
sys.path.append('..')
from utils.utils import mkdirs

from args import Parser
from nn.denseEDcirc2d import DenseED
from nn.bayesNN import BayesNN
from nn.swag import SwagNN
from utils.utils import mkdirs, toNumpy, toTuple
from utils.burgerLoader2D import BurgerLoader

import matplotlib as mpl
mpl.use('agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import rc

import torch
import numpy as np
import os, time

def testMSE(args, model, test_loader, tstep=100, test_every=2):
    '''
    Tests the base deterministic model of a single mini-batch
    Args:
        args (argparse): object with programs arguements
        model (PyTorch model): DenseED model to be tested
        test_loader (dataloader): dataloader with test cases (use createTestingLoader)
        tstep (int): number of timesteps to predict for
        test_every (int): Time-step interval to test (must match simulator), default = 2
    Returns:
        uPred (torch.Tensor): [d x t x n] predicted quantities
        u_target (torch.Tensor): [d x t x n] target/ simulated values
        mse_error  (torch.Tensor): [d x t] time dependednt mean squared error
        ese_error  (torch.Tensor): [d x t] time dependednt energy squared error
    '''
    model.eval()
    testIdx = (tstep)//test_every + 1
    mb_size = int(len(test_loader.dataset)/len(test_loader))
    u_out = torch.zeros(len(test_loader.dataset), tstep+1, 2, args.nel, args.nel)
    u_target = torch.zeros(len(test_loader.dataset), testIdx, 2, args.nel, args.nel)
    error = torch.zeros(len(test_loader.dataset), tstep+1, 2)
    error2 = torch.zeros(len(test_loader.dataset), tstep+1, 2)

    for bidx, (input0, uTarget0) in enumerate(test_loader):

        u_out[bidx*mb_size:(bidx+1)*mb_size, 0] = input0.cpu()
        u_target[bidx*mb_size:(bidx+1)*mb_size] = uTarget0[:,:testIdx].cpu()

        # Expand input to match model in channels
        dims = torch.ones(len(input0.shape))
        dims[1] = args.nic
        input = input0.repeat(toTuple(toNumpy(dims).astype(int))).to(args.device)

        # Auto-regress
        for tidx in range(tstep):
            uPred = model(input[:,-2*args.nic:,:,:])
            u_out[bidx*mb_size:(bidx+1)*mb_size, tidx+1] = uPred.detach().cpu()
            
            input = input[:,-2*int(args.nic-1):,:].detach()
            input0 = uPred.detach()
            input = torch.cat([input,  input0], dim=1)

    # Reshape and compress last three dims for easy mean calculations
    u_out = u_out.view(len(test_loader.dataset), tstep+1, -1)[:,::test_every]
    u_target = u_target.view(len(test_loader.dataset), testIdx, -1)
    # Calc MSE and ESE errors of all collocation points and x/y components
    mse_error = torch.mean(torch.pow(u_out - u_target, 2), dim=-1)
    ese_error = torch.pow(torch.sum(torch.pow(u_out, 2)/2.0, dim=-1)/(args.nel**2) - torch.sum(torch.pow(u_target, 2)/2.0, dim=-1)/(args.nel**2), 2)

    return u_out, u_target, mse_error, ese_error

def testBayesianMSE(args, swag_nn, test_loader, tstep=100, n_samples=10, test_every=2):
    '''
    Tests samples of the model using SWAG and calculates error values
    Args:
        args (argparse): object with programs arguements
        model (PyTorch model): DenseED model to be tested
        test_loader (dataloader): dataloader with test cases with no shuffle (use createTestingLoader)
        tstep (int): number of timesteps to predict for
        n_samples (int): Number of model samples to test
        test_every (int): Time-step interval to test (must match simulator), default = 2
    Returns:
        uPred (torch.Tensor): [d x nsamp x t x n] predicted quantities
        u_target (torch.Tensor): [d x nsamp x t x n] target/ simulated values
        mse_error  (torch.Tensor): [d x t] time dependednt mean squared error using expected prediction
        ese_error  (torch.Tensor): [d x t] time dependednt energy squared error using expected prediction
    '''
    testIdx = (tstep)//test_every + 1
    mb_size = int(len(test_loader.dataset)/len(test_loader))
    u_out = torch.zeros(len(test_loader.dataset), n_samples, tstep+1, 2, args.nel, args.nel)
    u_target = torch.zeros(len(test_loader.dataset), testIdx, 2, args.nel, args.nel)
    
    for i in range(n_samples):
        print('Executing model sample {:d}'.format(i))
        model = swag_nn.sample(diagCov=False)
        model.eval()

        for bidx, (input0, uTarget0) in enumerate(test_loader):
            u_out[bidx*mb_size:(bidx+1)*mb_size, i, 0] = input0
            u_target[bidx*mb_size:(bidx+1)*mb_size] = uTarget0[:,:testIdx]

            # Expand input to match model in channels
            dims = torch.ones(len(input0.shape))
            dims[1] = args.nic
            input = input0.repeat(toTuple(toNumpy(dims).astype(int))).to(args.device)

            # Auto-regress
            for tidx in range(tstep):
                uPred = model(input[:,-2*args.nic:,:,:])

                u_out[bidx*mb_size:(bidx+1)*mb_size, i, tidx+1] = uPred.detach().cpu()

                input = input[:,-2*int(args.nic-1):,:].detach()
                input0 = uPred.detach()
                input = torch.cat([input,  input0], dim=1)
            

    # Calc MSE and ESE errors
    u_out = u_out.view(len(test_loader.dataset), n_samples, tstep+1, -1)[:,:,::test_every]
    u_target = u_target.view(len(test_loader.dataset), testIdx, -1)
    mean_pred = torch.mean(u_out, dim=1)
    mse_error = torch.mean(torch.pow(mean_pred.double() - u_target.double(), 2), dim=-1)
    ese_error = torch.pow(torch.sum(torch.pow(mean_pred, 2)/2.0, dim=-1)/(args.nel**2)  - torch.sum(torch.pow(u_target, 2)/2.0, dim=-1)/(args.nel**2), 2)

    return u_out, u_target, mse_error, ese_error


def plotError(tT, mse_errors, energy_errors, titles):
    '''
    Plots MSE and ESE of test cases
    '''
    plt.close("all")
    # Create figure
    mpl.rcParams['font.family'] = ['serif'] # default is sans-serif
    # rc('text', usetex=True)

    fig = plt.figure(figsize=(16, 5))
    ax = []
    ax.append(plt.subplot2grid((1, 11), (0, 0), colspan=5))
    ax.append(plt.subplot2grid((1, 11), (0, 6), colspan=5))

    line_styles = ['-', '-.']
    for i in range(len(mse_errors)):
        # MSE: Average over test cases
        error_median = np.median(mse_errors[i], axis=0)
        error_mean = np.mean(mse_errors[i], axis=0)
        ax[0].plot(tT, error_median, line_styles[i], c='b', zorder=5, label='{} Median'.format(titles[i]), linewidth=2.0)
        ax[0].plot(tT, error_mean, line_styles[i], c='r', zorder=5, label='{} Mean'.format(titles[i]), linewidth=2.0)

        # ESE: Average over test cases
        error_median = np.median(energy_errors[i], axis=0)
        error_mean = np.mean(energy_errors[i], axis=0)
        ax[1].plot(tT, error_median, line_styles[i], c='b', zorder=5, label='{} Median'.format(titles[i]), linewidth=2.0)
        ax[1].plot(tT, error_mean, line_styles[i], c='r', zorder=5, label='{} Mean'.format(titles[i]), linewidth=2.0)

    ax[0].set_xlabel('t', fontsize=14)
    ax[0].set_ylabel('MSE', fontsize=14)
    ax[0].legend(loc='upper left')

    ax[1].set_xlabel('t', fontsize=14)
    ax[1].set_ylabel('ESE', fontsize=14)

    file_dir = '.'
    # If directory does not exist create it
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)
    file_name = file_dir+"/burger2D_MSE"
    plt.savefig(file_name+".png", bbox_inches='tight')
    plt.savefig(file_name+".pdf", bbox_inches='tight')

    # plt.show()

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
    test_cases = np.array([0, 1, 2]).astype(int)
    test_cases = np.arange(0, 200, 1).astype(int)
    testing_loader = burgerLoader.createTestingLoader('../solver/fenics_data', test_cases, simdt=0.005, batch_size=10)

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
    
    n_test = 200
    # Load the model
    swag_nn.loadModel(100, file_dir='./networks')
    with torch.no_grad():
        # Predict and calculate error values
        u_out, u_target, error, error2 = testMSE(args, swag_nn.base, testing_loader, tstep=n_test)

    # Load Bayesian network
    swag_nn = SwagNN(args, bayes_nn, full_cov=True, max_models=30)
    swag_nn.loadModel(200, file_dir="./networks")   
    with torch.no_grad():
        # Predict and calculate error values
        uPred, uTarget, bayes_error, bayes_error2 = testBayesianMSE(args, swag_nn, testing_loader, tstep=n_test, n_samples=30)


    tTest = np.arange(0, n_test*args.dt+1e-8, 2*args.dt)
    xTest = np.linspace(x0, x1, args.nel+1)

    mse_error = [error.cpu().numpy(), bayes_error.cpu().numpy()]
    energy_error = [error2.cpu().numpy(), bayes_error2.cpu().numpy()]
    titles = ['AR-DenseED', 'BAR-DenseED']
    plotError(tTest, mse_error, energy_error, titles)
