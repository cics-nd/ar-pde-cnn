import torch
import matplotlib as mpl
mpl.use('agg')

import numpy as np
import os
import scipy.integrate as integrate
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.lines import Line2D
from matplotlib import rc

def plotPred(args, t, xT, uPred, uTarget, epoch, bidx=0):
    '''
    Plots a single prediction contour
    '''
    plt.close("all")

    # Create figure
    mpl.rcParams['font.family'] = ['serif'] # default is sans-serif
    rc('text', usetex=False)

    fig = plt.figure(figsize=(15, 8), dpi=150)
    ax = []
    ax.append(plt.subplot2grid((3, 15), (0, 0), colspan=14))
    ax.append(plt.subplot2grid((3, 15), (1, 0), colspan=14))
    ax.append(plt.subplot2grid((3, 15), (2, 0), colspan=14))

    cmap = "inferno"
    c0 = ax[1].imshow(uPred.T, interpolation='nearest', cmap=cmap, aspect='auto', extent=[t[0],t[-1],xT[0],xT[-1]])
    c_max = np.max(uPred.T)
    c_min = np.min(uPred.T)
    c0.set_clim(vmin=c_min, vmax=c_max)

    c0 = ax[0].imshow(uTarget.T, interpolation='nearest', cmap=cmap, aspect='auto', extent=[t[0],t[-1],xT[0],xT[-1]])
    c0.set_clim(vmin=c_min, vmax=c_max)

    p0 = ax[0].get_position().get_points().flatten()
    p1 = ax[1].get_position().get_points().flatten()
    ax_cbar = fig.add_axes([p1[2]+0.015, p1[1], 0.020, p0[3]-p1[1]])
    ticks = np.linspace(0, 1, 5)
    tickLabels = np.linspace(c_min, c_max, 5)
    tickLabels = ["{:02.2f}".format(t0) for t0 in tickLabels]
    cbar = mpl.colorbar.ColorbarBase(ax_cbar, cmap=cmap, orientation='vertical', ticks=ticks)
    cbar.set_ticklabels(tickLabels)

    cmap = "viridis"
    c0 = ax[2].imshow(np.abs(uPred.T - uTarget.T), interpolation='nearest', cmap=cmap, aspect='auto', extent=[t[0],t[-1],xT[0],xT[-1]])
    p0 = ax[2].get_position().get_points().flatten()
    ax_cbar = fig.add_axes([p0[2]+0.015, p0[1], 0.020, p0[3]-p0[1]])
    ticks = np.linspace(0, 1, 5)
    tickLabels = np.linspace(c0.norm.vmin, c0.norm.vmax, 5)
    tickLabels = ["{:.2e}".format(t0) for t0 in tickLabels]
    cbar = mpl.colorbar.ColorbarBase(ax_cbar, cmap=cmap, orientation='vertical', ticks=ticks)
    cbar.set_ticklabels(tickLabels)

    ax[0].set_ylabel('x', fontsize=14)
    ax[1].set_ylabel('x', fontsize=14)
    ax[2].set_ylabel('x', fontsize=14)
    ax[2].set_xlabel('t', fontsize=14)

    file_name = args.pred_dir+"/burgerPred-epoch{0:03d}-{1:01d}.png".format(epoch, bidx)
    plt.savefig(file_name, bbox_inches='tight')

def plotSamples(args, t, xT, uPred, uTarget, epoch=0):
    '''
    Plots prediction contour of Baysian model samples
    '''
    plt.close("all")
    # Create figure
    mpl.rcParams['font.family'] = ['serif'] # default is sans-serif
    # rc('text', usetex=True)

    n_sample = uPred.shape[0] + 1
    nrow = int(np.sqrt(n_sample))
    ncol = 6*nrow + 1
    fig = plt.figure(figsize=(20, 10), dpi=150)
    ax = []
    for i in range(nrow):
        for j in range(nrow):
            ax.append(plt.subplot2grid((nrow, ncol), (i, 6*j), colspan=5))

    cmap = "inferno" 
    # Target in top left
    uTarget = uTarget[:uPred.shape[1]]
    c0 = ax[0].imshow(uTarget.T, interpolation='nearest', cmap=cmap, aspect='auto', extent=[t[0],t[-1],xT[0],xT[-1]])
    c_max = np.max(uPred.T)
    c_min = np.min(uPred.T)
    c0.set_clim(vmin=c_min, vmax=c_max)

    # Prediction samples
    for i in range(1, len(ax)):
        c0 = ax[i].imshow(uPred[i-1].T, interpolation='nearest', cmap=cmap, aspect='auto', extent=[t[0],t[-1],xT[0],xT[-1]])
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
        ax[i].set_xlabel('t')
    for i in range(nrow):
        ax[int(i*nrow)].set_ylabel('x')

    file_name = args.pred_dir+"/burgerSamples_epoch{:03d}.png".format(epoch)
    plt.savefig(file_name, bbox_inches='tight')

def calcR2score(uPred, uTarget, epoch=0, save=True):
    '''
    Calculates the total and time dependent average R2 score
    Args:
        uPred (torch.Tensor): [b x t x d] tensor of model predictions
        uTarget (torch.Tensor): [b x t x d] tensor of corresponding target values
        epoch (int): current training epoch (for logging)
    '''
    # Following:
    # https://en.wikipedia.org/wiki/Coefficient_of_determination
    # First total average
    ybar = torch.mean(uTarget.view(uTarget.size(0),-1), dim=-1)
    ss_tot = torch.sum(torch.pow(uTarget - ybar.unsqueeze(-1).unsqueeze(-1), 2).view(uTarget.size(0), -1), dim=-1)
    ss_res = torch.sum(torch.pow(uTarget - uPred, 2).view(uTarget.size(0), -1), dim=-1)

    r2_avg = torch.mean(1 - ss_res/ss_tot).cpu().numpy()

    # Now time dependent
    ybar = torch.mean(uTarget, dim=-1)
    ss_tot = torch.sum(torch.pow(uTarget - ybar.unsqueeze(-1), 2), dim=-1)
    ss_res = torch.sum(torch.pow(uTarget - uPred, 2), dim=-1)

    r2_time = torch.mean(1 - ss_res/ss_tot, dim=0).cpu().numpy()

    if(save):
        f=open('r2score_time.dat','ab')
        np.savetxt(f, np.insert(r2_time, 0, epoch)[np.newaxis,:], delimiter=',')
        f.close()

        f=open('r2score.dat','ab')
        np.savetxt(f, np.insert(r2_avg, 0, epoch)[np.newaxis,:], delimiter=',')
        f.close()