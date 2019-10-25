"""
Sovling 2D viscous Burgers' equation with Fenics
===
Distributed by: Notre Dame CICS (MIT Liscense)
- Associated publication:
url: http://www.sciencedirect.com/science/article/pii/S0021999119307612
doi: https://doi.org/10.1016/j.jcp.2019.109056
github: https://github.com/cics-nd/ar-pde-cnn
===
"""
import sys
sys.path.append("..") # Adds higher directory to python modules path.

import dolfin as df 
import numpy as np
import multiprocessing as mp
import matplotlib.pyplot as plt
import matplotlib
import time
from utils.utils import mkdirs
import scipy.io
import sys
matplotlib.use('agg')


class PeriodicBoundary(df.SubDomain):
    # https://fenicsproject.org/qa/262/possible-specify-more-than-one-periodic-boundary-condition/
    # Left boundary is "target domain" G
    def inside(self, x, on_boundary):
        # return True if on left or bottom boundary AND NOT on one of the two corners (0, 1) and (1, 0)
        return bool((df.near(x[0], 0) or df.near(x[1], 0,)) and 
                (not ((df.near(x[0], 0) and df.near(x[1], 1)) or 
                        (df.near(x[0], 1) and df.near(x[1], 0)))) and on_boundary)

    def map(self, x, y):
        if df.near(x[0], 1) and df.near(x[1], 1):
            y[0] = x[0] - 1.
            y[1] = x[1] - 1.
        elif df.near(x[0], 1):
            y[0] = x[0] - 1.
            y[1] = x[1]
        else:   # df.near(x[1], 1)
            y[0] = x[0]
            y[1] = x[1] - 1.


def init_field_fenics(mesh, V, order=4, seed=0):
    # https://fenicsproject.org/qa/3975/interpolating-vector-function-from-python-code-to-fenics/

    u0 = df.Function(V)
    # Extract x and y coordinates of mesh and
    # align with dof structure
    dim = V.dim()
    N = mesh.geometry().dim()
    coor = V.tabulate_dof_coordinates().reshape(dim, N)
    f0_dofs = V.sub(0).dofmap().dofs()
    f1_dofs = V.sub(1).dofmap().dofs()

    x = coor[:, 0]   # x for fx and fy
    y = coor[:, 1]   # y for fx and fy
    f0_x, f0_y = x[f0_dofs], y[f0_dofs]  # x, y of components
    # f1_x, f1_y = x[f1_dofs], y[f1_dofs]

    np.random.seed(seed)
    lam = np.random.randn(2, 2, (2*order+1)**2)
    c = -1 + np.random.rand(2) * 2
    aa, bb = np.meshgrid(np.arange(-order, order+1), np.arange(-order, order+1))
    k = np.stack((aa.flatten(), bb.flatten()), 1)
    kx_plus_ly = (np.outer(k[:, 0], f0_x) + np.outer(k[:, 1], f0_y))*2*np.pi

    # vector field
    f = np.dot(lam[0], np.cos(kx_plus_ly)) + np.dot(lam[1], np.sin(kx_plus_ly))
    f = f * 2 / np.amax(np.abs(f), axis=1, keepdims=True) + c[:, None]

    # Insert values of fx and fy into the function fe
    u0.vector()[f0_dofs] = f[0]
    u0.vector()[f1_dofs] = f[1]

    return u0, lam, c


def burgers2d(run, nu, ngx, ngy, dt, T, ngx_out, ngy_out, save_dir,
    save_every, save_pvd=False, save_vector=False, plot=False, order=4):
    """simulate 2D Burgers' equation
    https://www.firedrakeproject.org/demos/burgers.py.html

    Args:
        run (int): # run
        nu (float): viscosity
        ngx (int): # grid in x axis
        ngy (int):
        dt (float): time step for simulation
        T (float): simulation time from 0 to T
        ngx_out (int): output # grid in x axis
        ngy_out (int): output # grid in y axis
        save_dir (str): runs folder
        order (int): order for sampling initial U
        save_every (int): save frequency in terms of # dt
        save_pvd (bool): save the field as vtk file for paraview
        save_vector (bool): save fenics field vector for later operation
        plot (bool): plot fields
    """
    assert not (save_pvd and save_vector), 'wasting memory to save pvd & vector'
    save_dir = save_dir + f'/run{run}'
    mkdirs(save_dir)
    mesh = df.UnitSquareMesh(ngx-1, ngy-1)
    mesh_out = df.UnitSquareMesh(ngx_out-1, ngy_out-1)
    V = df.VectorFunctionSpace(mesh, 'CG', 2, constrained_domain=PeriodicBoundary())
    Vout = df.VectorFunctionSpace(mesh_out, 'CG', 1, constrained_domain=PeriodicBoundary())

    # initial vector field
    u0, lam, c = init_field_fenics(mesh, V, order=order, seed=run)
    np.savez(save_dir + '/init_lam_c.npz', lam=lam, c=c)

    u = df.Function(V)
    u_old = df.Function(V)
    v = df.TestFunction(V)

    u = df.project(u0, V)
    u_old.assign(u)

    # backward Euler
    F = (df.inner((u - u_old)/dt, v) \
        + df.inner(df.dot(u, df.nabla_grad(u)), v) \
        + nu*df.inner(df.grad(u), df.grad(v)))*df.dx

    t = 0
    k = 0
    vtkfile = df.File(save_dir + f'/soln{ngx_out}x{ngy_out}_.pvd')
    u_out = df.project(u, Vout)
    u_out.rename('u', 'u')
    # (2, ngy_out, ngx_out) ?
    u_out_vertex = u_out.compute_vertex_values(mesh_out).reshape(2, ngx_out, ngy_out)
    np.save(save_dir + f'/u{k}.npy', u_out_vertex)
    # if plot:
    #     plot_row([u_out_vertex[0], u_out_vertex[1]], save_dir, f'u{k}', 
    #         same_range=False, plot_fn='imshow', cmap='jet')
    if save_pvd:
        vtkfile << (u_out, t)
    elif save_vector:
        u_out_vector = u_out.vector().get_local()
        np.save(save_dir + f'/u{k}_fenics_vec.npy', u_out_vector)
    
    # u_vec_load = np.load(save_dir + f'/u{k}.npy')
    # u_load = Function(Vout)
    # u_load.vector().set_local(u_vec_load)

   # not much log
    df.set_log_level(30)
    tic = time.time()

    while t < T:

        t += dt
        k += 1
        df.solve(F == 0, u)
        u_old.assign(u)
        
        u_out = df.project(u, Vout)
        u_out.rename('u', 'u')

        if k % save_every == 0:
            u_out_vertex = u_out.compute_vertex_values(mesh_out).reshape(2, ngx_out, ngy_out)
            np.save(save_dir + f'/u{k}.npy', u_out_vertex)
            # if k % (10 * save_every) == 0 and plot:
            #     plot_row([u_out_vertex[0], u_out_vertex[1]], save_dir, f'u{k}', 
            #         same_range=False, plot_fn='imshow', cmap='jet')
        if save_pvd:
            vtkfile << (u_out, t)
        elif save_vector:
            u_out_vector = u_out.vector().get_local()
            np.save(save_dir + f'/u{k}_fenics_vec.npy', u_out_vector)

        print(f'Run {run}: solved {k} steps with total {time.time()-tic:.3f} seconds')

    return time.time() - tic



def burgers2d_mp(istart, iend, processes=12):
    
    # multiprocessing!
    ngx = 128
    ngy = 128
    ngx_out = 64
    ngy_out = 64
    nu = 0.005
    # dt should be small to ensure the stability
    dt = 0.005
    T = 1.0
    save_dt = 0.01
    # save every 0.01 s
    save_every = int(save_dt / dt)
    order = 4
    save_pvd = False
    save_vector = True
    plot = False
    save_dir = './fenics_data'

    pool = mp.Pool(processes=processes)
    print(f'Initialized pool with {processes} processes')
    results = [pool.apply_async(burgers2d, args=(run, nu, ngx, ngy, dt, T, ngx_out, 
        ngy_out, save_dir, save_every, save_pvd, save_vector, plot, order)) for run in range(istart, iend)]
    time_taken = [p.get() for p in results]
    print(time_taken)


if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser(description='Sim 2D Burgers equation')
    parser.add_argument('--istart', type=int, default=0, help='start index (default: 1)')
    parser.add_argument('--iend', type=int, default=12, help='start index (default: 12)')
    parser.add_argument('--processes', type=int, default=4, help='# processes (default: 12)')
    args = parser.parse_args()

    burgers2d_mp(args.istart, args.iend, args.processes)
