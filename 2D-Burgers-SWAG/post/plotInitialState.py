'''
This script plots samples of the initial state used in
the 2D coupled Burgers' system. This graphic can be seen
in Figure 16 of the paper.
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

import os
import numpy as np

def getInitialState(index, ncells, order=4):
    '''
    Generates an initial state for the 2D Burgers' system
    Args:
        index (int): index for random seed
        ncells (int): domain discretization
        order (int): order of the random Fourier series, default is 4
    '''
    np.random.seed(index+100000)
    x = np.linspace(0, 1, ncells+1)[:-1]
    xx, yy = np.meshgrid(x, x)
    aa, bb = np.meshgrid(np.arange(-order, order+1), np.arange(-order, order+1))
    k = np.stack((aa.flatten(), bb.flatten()), 1)
    kx_plus_ly = (np.outer(k[:, 0], xx.flatten()) + np.outer(k[:, 1], yy.flatten()))*2*np.pi

    lam = np.random.randn(2, 2, (2*order+1)**2)
    c = -1 + np.random.rand(2) * 2

    f = np.dot(lam[0], np.cos(kx_plus_ly)) + np.dot(lam[1], np.sin(kx_plus_ly))
    f = 2 * f / np.amax(np.abs(f), axis=1, keepdims=True) + c[:, None]

    return f.reshape(-1, ncells, ncells)

def getInitialState2(index, ncells, order=4):
    '''
    Non-batch version that can be used to better understand the math
    '''
    np.random.seed(index+100000)

    lam = np.random.randn(2, 2, (2*order+1)**2).reshape(2,2,2*order+1,2*order+1)
    c = -1 + np.random.rand(2) * 2
    x = np.linspace(0, 1, ncells+1)[:-1]
    X= np.zeros((2, ncells, ncells))

    for i, x0 in enumerate(x):
        for j, y0 in enumerate(x):            
            for k, k0 in enumerate(np.arange(-order,order+1,1)):
                for l, k1 in enumerate(np.arange(-order,order+1,1)):
                    X[0, i, j] += lam[0,0,k,l]*np.cos(2*np.pi*(k0*x0 + k1*y0)) + lam[1,0,k,l]*np.sin(2*np.pi*(k0*x0 + k1*y0))
                    X[1, i, j] += lam[0,1,k,l]*np.cos(2*np.pi*(k0*x0 + k1*y0)) + lam[1,1,k,l]*np.sin(2*np.pi*(k0*x0 + k1*y0))

    X = X.reshape((2,-1))
    f = 2 * X / np.amax(np.abs(X), axis=1, keepdims=True) + c[:, None]

    return f.reshape(-1, ncells, ncells)

if __name__ == '__main__':

    # Create figure
    mpl.rcParams['font.family'] = ['times new roman'] # default is sans-serif
    rc('text', usetex=True)
    cmap = "plasma"

    ncases = 5
    fig, ax = plt.subplots(2, ncases, figsize=(14, 5))
    cmin = -3
    cmax = 3
    for i in range(ncases):
        u = getInitialState(i, 64)

        ax[0,i].imshow(u[0], interpolation='nearest', cmap=cmap, origin='lower', aspect='auto', extent=[0,1,0,1], vmin=cmin, vmax=cmax)
        ax[1,i].imshow(u[1], interpolation='nearest', cmap=cmap, origin='lower', aspect='auto', extent=[0,1,0,1], vmin=cmin, vmax=cmax)

        ax[0,i].set_xticks([0, 0.25, 0.5, 0.75, 1.0])
        ax[0,i].set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        ax[1,i].set_xticks([0, 0.25, 0.5, 0.75, 1.0])
        ax[1,i].set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        ax[0,i].set_xticklabels([])
        if(i > 0):
            for ax0 in ax[:,i]:
                ax0.set_yticklabels([])
        else:
            ax[0,i].set_ylabel('y', fontsize=14)
            ax[1,i].set_ylabel('y', fontsize=14)

        ax[1,i].set_xlabel('x', fontsize=14)

    ax[0,2].set_title('u', fontsize=14)
    ax[1,2].set_title('v', fontsize=14)

    p0 = ax[0,-1].get_position().get_points().flatten()
    p1 = ax[1,-1].get_position().get_points().flatten()
    ax_cbar = fig.add_axes([p1[2]+0.01, p1[1], 0.020, p0[3]-p1[1]])
    ticks = np.linspace(0, 1, 5)
    tickLabels = np.linspace(cmin, cmax, 5)
    tickLabels = ["{:02.2f}".format(t0) for t0 in tickLabels]
    cbar = mpl.colorbar.ColorbarBase(ax_cbar, cmap=plt.get_cmap(cmap), orientation='vertical', ticks=ticks)
    cbar.set_ticklabels(tickLabels)

    file_dir = '.'
    # If director does not exist create it
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)
    file_name = file_dir+"/burger2D_initialState"

    plt.savefig(file_name+".png", bbox_inches='tight')
    plt.savefig(file_name+".pdf", bbox_inches='tight')

    plt.show()