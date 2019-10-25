import matplotlib as mpl
mpl.use('agg')

import numpy as np
import os
import scipy.integrate as integrate
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.lines import Line2D
from matplotlib import rc

def plotPred(args, t, xT, uPred, uTarget, case=0, epoch=0, sample=0):
    '''
    Plots a single prediction contour
    '''
    plt.close("all")
    # Create figure
    mpl.rcParams['font.family'] = ['serif'] # default is sans-serif
    # rc('text', usetex=True)

    fig = plt.figure(figsize=(16, 10), dpi=150)
    ax = []
    ax.append(plt.subplot2grid((3, 15), (0, 0), colspan=14))
    ax.append(plt.subplot2grid((3, 15), (1, 0), colspan=14))
    ax.append(plt.subplot2grid((3, 15), (2, 0), colspan=14))

    cmap = "rainbow" 
    #Target
    c0 = ax[0].imshow(uTarget[case].T, interpolation='nearest', cmap=cmap, aspect='auto', extent=[t[0],t[-1],xT[0],xT[-1]])
    c0.set_clim(vmin=-3, vmax=3)

    # Prediction
    c0 = ax[1].imshow(uPred[case].T, interpolation='nearest', cmap=cmap, aspect='auto', extent=[t[0],t[-1],xT[0],xT[-1]])
    c0.set_clim(vmin=-3, vmax=3)

    p0 = ax[0].get_position().get_points().flatten()
    p1 = ax[1].get_position().get_points().flatten()
    ax_cbar = fig.add_axes([p1[2]+0.01, p1[1], 0.020, p0[3]-p1[1]])
    ticks = np.linspace(0, 1, 5)
    tickLabels = np.linspace(-3, 3, 5)
    tickLabels = ["{:02.2f}".format(t0) for t0 in tickLabels]
    cbar = mpl.colorbar.ColorbarBase(ax_cbar, cmap=cmap, orientation='vertical', ticks=ticks)
    cbar.set_ticklabels(tickLabels)

    cmap = "viridis"
    c0 = ax[2].imshow(np.abs(uPred[case].T - uTarget[case].T), interpolation='nearest', cmap=cmap, aspect='auto', extent=[t[0],t[-1],xT[0],xT[-1]])
    p0 = ax[2].get_position().get_points().flatten()
    ax_cbar = fig.add_axes([p0[2]+0.01, p0[1], 0.020, p0[3]-p0[1]])
    ticks = np.linspace(0, 1, 5)
    tickLabels = np.linspace(c0.norm.vmin, c0.norm.vmax, 5)
    tickLabels = ["{:.2e}".format(t0) for t0 in tickLabels]
    cbar = mpl.colorbar.ColorbarBase(ax_cbar, cmap=cmap, orientation='vertical', ticks=ticks)
    cbar.set_ticklabels(tickLabels)

    ax[0].set_ylabel('x')
    ax[1].set_ylabel('x')
    ax[2].set_ylabel('x')

    ax[2].set_xlabel('t')
    ax[0].set_title('Epoch: {0:03}'.format(epoch))


    file_name = args.pred_dir+"/ksPred_epoch{:03d}_case{:03d}.png".format(epoch, case)
    plt.savefig(file_name, bbox_inches='tight')

def plotSamples(args, t, xT, uPred, uTarget, case=0, epoch=0):
    '''
    Plots prediction contour of Baysian model samples
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
    c0 = ax[0].imshow(uTarget[case].T, interpolation='nearest', cmap=cmap, aspect='auto', extent=[t[0],t[-1],xT[0],xT[-1]])
    c0.set_clim(vmin=-3, vmax=3)

    # Prediction samples
    for i in range(1, len(ax)):
        c0 = ax[i].imshow(uPred[case, i-1].T, interpolation='nearest', cmap=cmap, aspect='auto', extent=[t[0],t[-1],xT[0],xT[-1]])
        c0.set_clim(vmin=-3, vmax=3)

    p0 = ax[nrow-1].get_position().get_points().flatten()
    p1 = ax[-1].get_position().get_points().flatten()
    ax_cbar = fig.add_axes([p1[2]+0.01, p1[1], 0.020, p0[3]-p1[1]])
    ticks = np.linspace(0, 1, 5)
    tickLabels = np.linspace(-3, 3, 5)
    tickLabels = ["{:02.2f}".format(t0) for t0 in tickLabels]
    cbar = mpl.colorbar.ColorbarBase(ax_cbar, cmap=cmap, orientation='vertical', ticks=ticks)
    cbar.set_ticklabels(tickLabels)

    # Axis labels
    for i in range(len(ax)-nrow, len(ax)):
        ax[i].set_xlabel('t')
    for i in range(nrow):
        ax[int(i*nrow)].set_ylabel('x')

    file_name = args.pred_dir+"/ksSamples_epoch{:03d}_case{:03d}.png".format(epoch, case)
    plt.savefig(file_name, bbox_inches='tight')
