'''
This script plots the average mean squared error (MSE) and 
energy squared error (ESE) as a function of time for a test 
set of 200 cases using AR-DenseED and predictive expectation 
of BAR-DenseED. The produced graphic is seen in Figure 11 of 
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

def testMSE(args, model, test_loader, tstep=100):
    '''
    Tests the base deterministic model and calculates error values
    Args:
        args (argparse): object with programs arguements
        model (PyTorch model): DenseED model to be tested
        test_loader (dataloader): dataloader with test cases (use createTestingLoader)
        tstep (int): number of timesteps to predict for
    Returns:
        uPred (torch.Tensor): [d x t x n] predicted quantities
        u_target (torch.Tensor): [d x t x n] target/ simulated values
        mse_error  (torch.Tensor): [d x t] time dependednt mean squared error
        ese_error  (torch.Tensor): [d x t] time dependednt energy squared error
    '''
    model.eval()
    mb_size = int(len(test_loader.dataset)/len(test_loader))
    u_out = torch.zeros(len(test_loader.dataset), tstep+1, args.nel).to(args.device)
    u_target = torch.zeros(len(test_loader.dataset), tstep+1, args.nel).to(args.device)
    error = torch.zeros(len(test_loader.dataset), tstep+1)
    error2 = torch.zeros(len(test_loader.dataset), tstep+1)

    for bidx, (input0, uTarget0) in enumerate(test_loader):
        input = input0.to(args.device)

        u_out[bidx*mb_size:(bidx+1)*mb_size, 0, :] = input[:,0]
        u_target[bidx*mb_size:(bidx+1)*mb_size] = uTarget0[:,:tstep+1]
        # Auto-regress
        for tidx in range(tstep):
            uPred = model(input[:,-args.nic:,:])
            u_out[bidx*mb_size:(bidx+1)*mb_size, tidx+1, :] = uPred[:,0]
            
            input = input[:,-(args.nic-1):,:].detach()
            input0 = uPred[:,0,:].unsqueeze(1).detach()
            input = torch.cat([input,  input0], dim=1)

        if(bidx%5 == 0):
            print('Executed Batch {}/{}'.format(bidx, len(test_loader)))

    # Calc MSE and ESE errors
    mse_error = torch.mean(torch.pow(u_out - u_target[:,:tstep+1], 2), dim=-1)
    ese_error = torch.pow(torch.sum(torch.pow(u_out, 2)/2.0, dim=-1)/args.nel - torch.sum(torch.pow(u_target[:,:tstep+1], 2)/2.0, dim=-1)/args.nel, 2)

    return u_out, u_target, mse_error, ese_error

def testBayesianMSE(args, swag_nn, test_loader, tstep=100, n_samples=10):
    '''
    Tests samples of the model using SWAG and calculates error values
    Args:
        args (argparse): object with programs arguements
        model (PyTorch model): DenseED model to be tested
        test_loader (dataloader): dataloader with test cases with no shuffle (use createTestingLoader)
        tstep (int): number of timesteps to predict for
        n_samples (int): Number of model samples to test
    Returns:
        uPred (torch.Tensor): [d x nsamp x t x n] predicted quantities
        u_target (torch.Tensor): [d x nsamp x t x n] target/ simulated values
        mse_error  (torch.Tensor): [d x t] time dependednt mean squared error using expected prediction
        ese_error  (torch.Tensor): [d x t] time dependednt energy squared error using expected prediction
    '''
    mb_size = int(len(test_loader.dataset)/len(test_loader))
    u_out = torch.zeros(len(test_loader.dataset), n_samples, tstep+1, args.nel)
    u_target = torch.zeros(len(test_loader.dataset), tstep+1, args.nel)
    
    for i in range(n_samples):
        print('Executing model sample {:d}'.format(i))
        model = swag_nn.sample(diagCov=False)
        model.eval()

        for bidx, (input0, uTarget0) in enumerate(test_loader):
            input = input0.to(args.device)
            u_target[bidx*mb_size:(bidx+1)*mb_size] = uTarget0[:,:tstep+1]
            u_out[bidx*mb_size:(bidx+1)*mb_size,i, 0, :] = input[:,0]
            # Auto-regress
            for t_idx in range(tstep):
                uPred = model(input[:,-args.nic:,:])

                u_out[bidx*mb_size:(bidx+1)*mb_size, i, t_idx+1, :] = uPred[:,0]

                input = input[:,-int(args.nic-1):,:].detach()
                input0 = uPred[:,0,:].unsqueeze(1).detach()
                input = torch.cat([input,  input0], dim=1)
            
            if(bidx%5 == 0):
                print('Executed Batch {}/{}'.format(bidx+1, len(test_loader)))

    # Calc expected MSE and ESE errors
    mean_pred = torch.mean(u_out, dim=1)
    var_pred = torch.var(u_out, dim=1)
    print(var_pred.size())

    mse_error = torch.mean(torch.pow(mean_pred.double() - u_target[:,:tstep+1].double(), 2), dim=-1)
    ese_error = torch.pow(torch.sum(torch.pow(mean_pred, 2)/2.0, dim=-1)/args.nel - torch.sum(torch.pow(u_target[:,:tstep+1], 2)/2.0, dim=-1)/args.nel, 2)

    return u_out, u_target, mse_error, ese_error


def plotError(tT, mse_errors, energy_errors, titles):
    '''
    Plots MSE and ESE of test cases
    '''
    plt.close("all")
    # Create figure
    mpl.rcParams['font.family'] = ['serif'] # default is sans-serif
    rc('text', usetex=True)

    fig = plt.figure(figsize=(16, 5))
    ax = []
    ax.append(plt.subplot2grid((1, 11), (0, 0), colspan=5))
    ax.append(plt.subplot2grid((1, 11), (0, 6), colspan=5))

    line_styles = ['-', '-.']
    for i in range(len(mse_errors)):
        # MSE: Average over test cases
        error_median = np.median(mse_errors[i], axis=0)
        error_mean = np.mean(mse_errors[i], axis=0)
        error_std = np.std(mse_errors[i], axis=0)
        ax[0].plot(tT, error_median, line_styles[i], c='b', zorder=5, label='{} Median'.format(titles[i]), linewidth=2.0)
        ax[0].plot(tT, error_mean, line_styles[i], c='r', zorder=5, label='{} Mean'.format(titles[i]), linewidth=2.0)

        # ESE: Average over test cases
        error_median = np.median(energy_errors[i], axis=0)
        error_mean = np.mean(energy_errors[i], axis=0)
        error_std = np.std(energy_errors[i], axis=0)
        ax[1].plot(tT, error_median, line_styles[i], c='b', zorder=5, label='{} Median'.format(titles[i]), linewidth=2.0)
        ax[1].plot(tT, error_mean, line_styles[i], c='r', zorder=5, label='{} Mean'.format(titles[i]), linewidth=2.0)


    ax[0].set_ylim([0, 0.075])
    ax[0].set_xlim([-0.05, 2.0])
    ax[0].set_xlabel('t', fontsize=14)
    ax[0].set_ylabel('MSE', fontsize=14)
    ax[0].legend(loc='upper right')

    ax[1].set_ylim([0, 0.015])
    ax[1].set_xlim([-0.05, 2.0])
    ax[1].set_xlabel('t', fontsize=14)
    ax[1].set_ylabel('ESE', fontsize=14)

    file_dir = '.'
    # If directory does not exist create it
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)
    file_name = file_dir+"/burger_MSE"
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
    test_cases = np.arange(0,200,1).astype(int)
    testing_loader = burgerLoader.createTestingLoader('../solver/fenics_data_dt0.001_T2.0', test_cases, dt=0.001, batch_size=25)

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
    
    # Load deterministic network
    swag_nn.loadModel(100, file_dir="./networks")
    with torch.no_grad():
        u_out, u_target, error, error2 = testMSE(args, swag_nn.base, testing_loader, tstep=400)

    # Load Bayesian network
    swag_nn.loadModel(200, file_dir="./networks")
    with torch.no_grad():
        uPred, uTarget, bayes_error, bayes_error2 = testBayesianMSE(args, swag_nn, testing_loader, tstep=400, n_samples=30)

    tTest = np.arange(0, 400*args.dt+1e-8, args.dt)
    xTest = np.linspace(x0, x1, args.nel+1)

    mse_error = [error.cpu().numpy(), bayes_error.cpu().numpy()]
    energy_error = [error2.cpu().numpy(), bayes_error2.cpu().numpy()]
    titles = ['AR-DenseED', 'BAR-DenseED']
    plotError(tTest, mse_error, energy_error, titles)