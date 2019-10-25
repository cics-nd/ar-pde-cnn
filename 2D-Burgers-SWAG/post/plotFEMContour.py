'''
This script plots a FEM simulations of the 2D coupled Burgers equation.
The produced graphic is seen in Figure 17 of the paper.
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

import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import rc
from matplotlib import cm
import matplotlib.ticker as ticker
from mpl_toolkits.axes_grid1 import make_axes_locatable

import numpy as np
import os

if __name__ == '__main__':

    # Create figure
    mpl.rcParams['font.family'] = ['times new roman'] # default is sans-serif
    rc('text', usetex=True)

    cmap = "plasma"
    # Case 18: Figure 17a, Case 19: Figure 17b
    case = 18
    dt = 0.005
    times = np.arange(0, 60, 10)

    # Read in the data
    data_dir = "../solver/fenics_data"
    case_dir = os.path.join(data_dir, "run{:d}".format(case))
    print("Reading test case: {}".format(case_dir))
    seq = []
    for j in times:
        file_dir = os.path.join(case_dir, "u{:d}.npy".format(j))
        u0 = np.load(file_dir)
        # Remove the periodic nodes
        seq.append(u0[:,:,:])

    u = np.stack(seq, axis=0)

    fig, ax = plt.subplots(2, len(times), figsize=(19, 5))
    fig.subplots_adjust(wspace=0.5)

    def fmt(x, pos):
        label = "{:.02f}".format(x)
        return str(label)

    for i in range(len(times)):
        caxu = ax[0, i].imshow(u[i, 0], interpolation='nearest', cmap=cmap, origin='lower', aspect='auto', extent=[0,1,0,1])
        c_max = np.max(u[i, 0])
        c_min = np.min(u[i, 0])

        p0 = ax[0, i].get_position().get_points().flatten()
        ax_cbar = fig.add_axes([p0[2]+0.005, p0[1], 0.005, p0[3]-p0[1]])
        ticks = np.linspace(0, 1, 5)
        tickLabels = np.linspace(c_min, c_max, 5)
        tickLabels = ["{:02.2f}".format(t0) for t0 in tickLabels]
        cbar = mpl.colorbar.ColorbarBase(ax_cbar, cmap=plt.get_cmap(cmap), orientation='vertical', ticks=ticks)
        cbar.set_ticklabels(tickLabels)

        caxv = ax[1, i].imshow(u[i, 1], interpolation='nearest', cmap=cmap, origin='lower', aspect='auto', extent=[0,1,0,1])
        c_max = np.max(u[i, 1])
        c_min = np.min(u[i, 1])
        p0 = ax[1, i].get_position().get_points().flatten()
        ax_cbar = fig.add_axes([p0[2]+0.005, p0[1], 0.005, p0[3]-p0[1]])
        ticks = np.linspace(0, 1, 5)
        tickLabels = np.linspace(c_min, c_max, 5)
        tickLabels = ["{:02.2f}".format(t0) for t0 in tickLabels]
        cbar = mpl.colorbar.ColorbarBase(ax_cbar, cmap=plt.get_cmap(cmap), orientation='vertical', ticks=ticks)
        cbar.set_ticklabels(tickLabels)

        
        ax[0,i].set_xticks([0, 0.25, 0.5, 0.75, 1.0])
        ax[0,i].set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        ax[1,i].set_xticks([0, 0.25, 0.5, 0.75, 1.0])
        ax[1,i].set_yticks([0, 0.25, 0.5, 0.75, 1.0])

        ax[0,i].set_title('t={:.02f}'.format(dt*times[i]))
        ax[0,i].set_xticklabels([])
        if(i > 0):
            for ax0 in ax[:,i]:
                ax0.set_yticklabels([])
        else:
            ax[0,i].set_ylabel('y', fontsize=14)
            ax[1,i].set_ylabel('y', fontsize=14)
        ax[1,i].set_xlabel('x', fontsize=14)

    file_dir = '.'
    # If directory does not exist create it
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)
    file_name = file_dir+"/burger2D_fenics{:d}".format(case)

    plt.savefig(file_name+".png", bbox_inches='tight')
    plt.savefig(file_name+".pdf", bbox_inches='tight')

    plt.show()   