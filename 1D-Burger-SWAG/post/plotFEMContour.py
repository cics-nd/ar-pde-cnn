'''
This script plots four FEM simulations of the 1D viscous Burgers equation.
The produced graphic is seen in Figure 9 of the paper.
===
Distributed by: Notre Dame CICS (MIT Liscense)
- Associated publication:
url: http://www.sciencedirect.com/science/article/pii/S0021999119307612
doi: https://doi.org/10.1016/j.jcp.2019.109056
github: https://github.com/cics-nd/ar-pde-cnn
===
'''
import matplotlib as mpl

import numpy as np
import os
import scipy.integrate as integrate
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.lines import Line2D
from matplotlib import rc

if __name__ == '__main__':

    plt.close("all")

    # Create figure
    mpl.rcParams['font.family'] = ['serif'] # default is sans-serif
    rc('text', usetex=False)

    fig = plt.figure(figsize=(15, 4), dpi=150)
    ax = []
    ax.append(plt.subplot2grid((9, 28), (0, 0), colspan=11, rowspan=4))
    ax.append(plt.subplot2grid((9, 28), (0, 14), colspan=11, rowspan=4))
    ax.append(plt.subplot2grid((9, 28), (5, 0), colspan=11, rowspan=4))
    ax.append(plt.subplot2grid((9, 28), (5, 14), colspan=11, rowspan=4))

    tstep =2000
    dt = 0.001
    x0 = 0
    x1= 1.0
    nel = 512
    t = np.arange(0, tstep*dt+1e-8, dt)
    x = np.linspace(x0, x1, nel+1)
    cmap = "inferno"

    data_dir = '../solver/fenics_data_dt0.001_T2.0'
    for i, val in enumerate(np.arange(14,18,1)):
        file_name = data_dir+"/u{:d}.npy".format(val)
        print("Reading file: {}".format(file_name))
        uPred = np.load(file_name)[:tstep,:]

        c0 = ax[i].imshow(uPred.T, interpolation='nearest', cmap=cmap, origin='lower', aspect='auto', extent=[t[0],t[-1],x[0],x[-1]])
        c_max = np.max(uPred.T)
        c_min = np.min(uPred.T)
        c0.set_clim(vmin=c_min, vmax=c_max)

        p0 = ax[i].get_position().get_points().flatten()
        ax_cbar = fig.add_axes([p0[2]+0.015, p0[1], 0.020, p0[3]-p0[1]])
        ticks = np.linspace(0, 1, 5)
        tickLabels = np.linspace(c0.norm.vmin, c0.norm.vmax, 5)
        tickLabels = ["{:.2f}".format(t0) for t0 in tickLabels]
        cbar = mpl.colorbar.ColorbarBase(ax_cbar, cmap=plt.get_cmap(cmap), orientation='vertical', ticks=ticks)
        cbar.set_ticklabels(tickLabels)

    ax[0].set_xticklabels([])
    ax[1].set_xticklabels([])
    ax[1].set_yticklabels([])
    ax[3].set_yticklabels([])

    ax[0].set_ylabel('x', fontsize=14)
    ax[2].set_ylabel('x', fontsize=14)
    ax[2].set_xlabel('t', fontsize=14)
    ax[3].set_xlabel('t', fontsize=14)

    file_dir = '.'
    # If directory does not exist create it
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)
    file_name = file_dir+"/burgerFenics"
    plt.savefig(file_name+".png", bbox_inches='tight')
    plt.savefig(file_name+".pdf", bbox_inches='tight')

    plt.show()