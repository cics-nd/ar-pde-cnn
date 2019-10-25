'''
This script plots two spectral simulations of the Kuramoto-Sivashinsky equation.
The produced graphic is seen in Figure 4 of the paper.
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
    '''
    Plots matlab simulation result
    '''
    plt.close("all")
    # Create figure
    mpl.rcParams['font.family'] = ['serif'] # default is sans-serif
    rc('text', usetex=True)

    fig = plt.figure(figsize=(12, 3), dpi=150)
    ax = []
    ax.append(plt.subplot2grid((1, 14), (0, 0), colspan=6))
    ax.append(plt.subplot2grid((1, 14), (0, 7), colspan=6))
    cmap = "rainbow"

    dt = 0.1

    for i, val in enumerate([4,5]):
        file_name = "../solver/ks_data_{:d}.dat".format(val)
        print("Reading file: {}".format(file_name))
        u = np.loadtxt(file_name, delimiter=',')
        uData = (u[:,:-1]+u[:,1:])/2.0 # TODO: fix this
        uData = uData[:int(300/dt),:]

        x0 = np.linspace(0, 22*np.pi, uData.shape[-1])
        print(uData.shape[0])
        t0 = np.linspace(0,dt*uData.shape[0], uData.shape[0])
        T, X = np.meshgrid(t0, x0)
        # c0 = ax[i].contourf(T, X, uData.T, 50, cmap=cmap)
        c0 = ax[i].imshow(uData.T, interpolation='nearest', cmap=cmap, origin='lower', aspect='auto', extent=[t0[0],t0[-1],x0[0],x0[-1]])
        c0.set_clim(vmin=-3, vmax=3)

    p0 = ax[1].get_position().get_points().flatten()
    ax_cbar = fig.add_axes([p0[2]+0.02, p0[1], 0.020, p0[3]-p0[1]])
    ticks = np.linspace(0, 1, 5)
    tickLabels = np.linspace(c0.norm.vmin, c0.norm.vmax, 5)
    tickLabels = ["{:02.2f}".format(t0) for t0 in tickLabels]
    cbar = mpl.colorbar.ColorbarBase(ax_cbar, cmap=plt.get_cmap(cmap), orientation='vertical', ticks=ticks)
    cbar.set_ticklabels(tickLabels)

    for ax0 in ax:
        ax0.set_xlabel('t', fontsize=14)
        ax0.set_ylabel('x', fontsize=14)

    file_name = "ks-simulation.pdf"
    plt.savefig(file_name, bbox_inches='tight')
    file_name = "ks-simulation.png"
    plt.savefig(file_name, bbox_inches='tight')
    plt.show()




