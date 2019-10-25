import torch
import matplotlib as mpl
mpl.use('agg')

import numpy as np
import os
import scipy.integrate as integrate
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D
from matplotlib import rc
from utils.utils import toNumpy

def plotPred(args, case, uPred, uTarget, tsteps=10, target_step=2, pred_step=2, epoch=0):
    '''
    Plots several timesteps of the 2D Burger system
    Args:
        args (argparse): object with programs arguements
        case (int): test case number to plot
        uPred (torch.Tensor): [d x t1 x n x n] predicted quantities
        uTarget (torch.Tensor): [d x t2 x n x n] target/simulated quantities
        tsteps (int): max time-step number to plot up to
        target_step, pred_step (int): Interval to plot pred and target data respectively
    '''
    plt.close("all")

    # Create figure
    mpl.rcParams['font.family'] = ['serif'] # default is sans-serif
    rc('text', usetex=False)

    # Kinda ghetto but it works
    target_steps = np.arange(1, tsteps, target_step).astype(int)
    pred_steps = np.arange(1, int(pred_step*tsteps/target_step), pred_step).astype(int)

    target = uTarget[target_steps]
    prediction = uPred[pred_steps]

    data = [target, prediction, prediction - target]
    titles = [f'T={t:.3f}' for t in pred_steps * args.dt]
    ylabels = ['u_sim', 'u_pred', '2-1', 'v_sim', 'v_pred', '5-4']

    print(f'epoch {epoch}, plot prediction {case}')
    fig, ax = plt.subplots(6, len(pred_steps), figsize=(len(pred_steps)*3.5, 6*3))
    for i_ax in range(6):
        for j_ax in range(len(pred_steps)):
            ax0 = ax[i_ax, j_ax]
            ax0.axis('off')
            cmap = 'viridis' if i_ax in [2, 5] else 'plasma'
            cax = ax0.imshow(data[i_ax%3][j_ax, i_ax//3], origin='lower', cmap=cmap)

            cbar = plt.colorbar(cax, ax=ax0, fraction=0.046, pad=0.04, 
                format=ticker.ScalarFormatter(useMathText=True))
            cbar.formatter.set_powerlimits((-2, 2))
            cbar.update_ticks()

            if i_ax == 0:
                ax0.set_title(titles[j_ax])
            if j_ax == 0:
                ax0.set_ylabel(ylabels[i_ax])
        
    plt.tight_layout()
    plt.savefig(args.pred_dir + f'/burgerPred2D_epoch{epoch}_pred{case}.png', bbox_inches='tight')
    plt.close()

def plotSamples(args, case, uPred, uTarget, tstep=1, epoch=0):
    '''
    Plots BAR-DenseED samples of one timestep
    Args:
        args (argparse): object with programs arguements
        case (int): test case number to plot
        uPred (torch.Tensor): [d x t1 x n x n] predicted quantities
        uTarget (torch.Tensor): [d x t2 x n x n] target/simulated quantities
        tstep (int): time-step to plot samples at
    '''
    plt.close("all")
    # Create figure
    mpl.rcParams['font.family'] = ['serif'] # default is sans-serif
    # rc('text', usetex=True)

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
        c0 = ax[0].imshow(uTarget, interpolation='nearest', cmap=cmap, aspect='auto', extent=[0,1,0,1])
        c_max = np.max(uPred.T)
        c_min = np.min(uPred.T)
        c0.set_clim(vmin=c_min, vmax=c_max)

        # Prediction samples
        for i in range(1, len(ax)):
            c0 = ax[i].imshow(uPred[i-1], interpolation='nearest', cmap=cmap, aspect='auto', extent=[0,1,0,1])
            c0.set_clim(vmin=c_min, vmax=c_max)

        p0 = ax[nrow-1].get_position().get_points().flatten()
        p1 = ax[-1].get_position().get_points().flatten()
        ax_cbar = fig.add_axes([p1[2]+0.01, p1[1], 0.020, p0[3]-p1[1]])
        ticks = np.linspace(0, 1, 5)
        tickLabels = np.linspace(c_min, c_max, 5)
        tickLabels = ["{:02.2f}".format(t0) for t0 in tickLabels]
        cbar = mpl.colorbar.ColorbarBase(ax_cbar, cmap=cmap, orientation='vertical', ticks=ticks)
        cbar.set_ticklabels(tickLabels)

        # Axis labels
        for i in range(len(ax)-nrow, len(ax)):
            ax[i].set_xlabel('y')
        for i in range(nrow):
            ax[int(i*nrow)].set_ylabel('x')

    plotGrid(fig, axu, uTarget[tstep, 0], uPred[:,tstep,0])
    plotGrid(fig, axv, uTarget[tstep, 1], uPred[:,tstep,1])
    plt.suptitle('T: {:03f}'.format(args.dt*tstep))
    file_name = args.pred_dir+"/burger2dSamples_epoch{:03d}_t{:03f}_{:d}.png".format(epoch, args.dt*tstep, case)
    plt.savefig(file_name, bbox_inches='tight')