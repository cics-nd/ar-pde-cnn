'''
Finite difference solver for the 1D viscous Burgers' equation implemented 
using PyTorch tensors for easy GPU execution.

Time integration: runge-kutta 4, 
grad: first-order upwind, 
div: 2nd order central difference
===
Distributed by: Notre Dame CICS (MIT Liscense)
- Associated publication:
url: http://www.sciencedirect.com/science/article/pii/S0021999119307612
doi: https://doi.org/10.1016/j.jcp.2019.109056
github: https://github.com/cics-nd/ar-pde-cnn
===
'''
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import utils.cpuinfo as cpuinfo
import argparse
import json
plt.switch_backend('agg')


def mkdirs(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def burgersFunc(central_diff_conv, back_fwd_diff_conv):

    def burgers_rhs(u):
        # u_t = - u*u_x + nu*u_xx
        # central difference for div
        u_xx = central_diff_conv(u)

        # upwind for grad
        u_x_back_fwd = back_fwd_diff_conv(u)
        u_x = torch.where(u > 0, u_x_back_fwd[:, [0]], u_x_back_fwd[:, [1]])

        return - u*u_x + args.nu*u_xx

    return burgers_rhs

def rk4_step(func, dt, u):
    # return the increment
    k1 = func(u)
    k2 = func(u + k1 * dt / 2)
    k3 = func(u + k2 * dt / 2)
    k4 = func(u + k3 * dt)

    return (k1 + 2 * k2 + 2 * k3 + k4) * dt / 6


def burgers1d(u0, dt, nsteps, grad_order=[1, 2], device="cpu"):

    # upwind for grad
    if grad_order[0] == 2:
        back_fwd_diff_conv = nn.Conv1d(2, 1, kernel_size=5, stride=1, bias=False,
            padding=2*2, padding_mode='circular').to(device)
        back_fwd_diff_conv.weight.data = torch.DoubleTensor(
            [[[1, -4, 3, 0, 0]], [[0, 0, -3, 4, -1]]]).to(device) / (dx*2)
    elif grad_order[0] == 1:
        back_fwd_diff_conv = nn.Conv1d(2, 1, kernel_size=3, stride=1, bias=False,
            padding=1*2, padding_mode='circular').to(device)
        back_fwd_diff_conv.weight.data = torch.DoubleTensor(
            [[[-1, 1, 0]], [[0, -1, 1]]]).to(device) / dx
    back_fwd_diff_conv.requires_grad = False

    # central difference for second derivative
    if grad_order[1] == 2:
        central_diff_conv = nn.Conv1d(1, 1, kernel_size=3, stride=1, bias=False,
            padding=1*2, padding_mode='circular').to(device)
        central_diff_conv.weight.data = torch.DoubleTensor([[[1, -2, 1]]]).to(device) / (dx**2)
    elif grad_order[1] == 4:
        central_diff_conv = nn.Conv1d(1, 1, kernel_size=5, stride=1, bias=False,
            padding=2*2, padding_mode='circular').to(device)
        central_diff_conv.weight.data = torch.DoubleTensor([[[-1/12, 4/3, -5/2, 4/3, -1/12]]]).to(device) / (dx**2)
    central_diff_conv.requires_grad = False

    u = torch.DoubleTensor(u0[0][None, None, :]).to(device)
    u_fd = torch.zeros(nsteps, u.size(-1)).to(device)
    burgers_rhs = burgersFunc(central_diff_conv, back_fwd_diff_conv)
    with torch.no_grad():
        tic = time.time()
        for i in range(nsteps):
            u = u + rk4_step(burgers_rhs, dt, u)
            u_fd[i]  = u[0,0]
            # u_fd.append(u)
        # u_fd = torch.cat(u_fd, -2)[0].detach().cpu().numpy()
        time_taken = time.time()-tic
        u_fd = u_fd.detach().cpu().numpy()
        print(f'Solved {idx}-th 1D Burgers for {nsteps} steps in {time_taken} seconds')
        print(u_fd.shape)

    return u_fd, time_taken

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Sim 1D Burgers equation with FD')
    parser.add_argument('--istart', type=int, default=0, help='start index (default: 1)')
    parser.add_argument('--iend', type=int, default=10, help='start index (default: 10)')
    args = parser.parse_args()

    args.dt = 0.0001
    args.T = 2.0
    args.ncell = 512
    args.nu = 0.0025
    nsteps = int(args.T /args.dt)
    dx = 1. / args.ncell

    div_central_diff_order = 2
    grad_up_wind_order = 1

    plot_every = 25

    # Where to read in the initial conditions
    data_root = './fenics_data_dt0.0005_T2.0'
    device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
    # device = 'cpu'
    if(device == 'cpu'):
        args.hardware = cpuinfo.cpu.info[0]['model name']
        print('CPU: {}'.format(args.hardware))
        args.save_dir = f'./fd_data_dt{args.dt}_T{args.T}'
    else:
        args.hardware = torch.cuda.get_device_name(0)
        args.save_dir = f'./fd_data_dt{args.dt}_T{args.T}_GPU'
    
    plot_dir = args.save_dir+'/fd_img'
    mkdirs(args.save_dir)
    mkdirs(plot_dir)
    
    with open(args.save_dir + "/args.txt", 'w') as args_file:
        json.dump(vars(args), args_file, indent=4)

    # Time-step loop
    times = []
    for idx in range(args.istart, args.iend):
        # Load FEM, needed so the initial conditions are the same!
        u_fem = np.load(data_root + f'/u{idx}.npy')[:, :-1]

        u_fd, time_taken = burgers1d(u_fem, args.dt, nsteps, device=device)
        times.append(np.array([idx,time_taken]))
        # Before saving add the periodic boundary
        u_fd0 = np.concatenate([u_fd, u_fd[:,:1]], axis=1)
        np.save(args.save_dir + f'/u{idx}.npy', u_fd0)

        n = nsteps // plot_every
        colors = plt.cm.rainbow(np.linspace(0, 1, n))
        for i in range(n):
            plt.plot(u_fd[i*plot_every], color=colors[i], linewidth=0.5)
        plt.savefig(plot_dir + f'/u{idx}_profile_steps{nsteps}_time{time_taken:.4f}.png')
        plt.close()

        fem_dt = 0.0005
        if(fem_dt >= args.dt):
            n = int(fem_dt/args.dt)
            data = [u_fd[::n], u_fem[1:1+nsteps//n], np.abs(u_fd[::n]-u_fem[1:1+nsteps//n])]
        else:
            n = int(args.dt/fem_dt)
            data = [u_fd, u_fem[1:1+n*nsteps:n], np.abs(u_fd-u_fem[1:1+n*nsteps:n])]

    times = np.stack(times, axis=0)
    np.save(args.save_dir + f'/ufdTimes_dt{args.dt}_T{args.T}.npy', times)