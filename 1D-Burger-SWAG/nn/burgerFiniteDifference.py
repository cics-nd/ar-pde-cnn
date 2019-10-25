import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair, _quadruple
import numpy as np
import matplotlib.pyplot as plt

class FourthOrderFilter1D(object):
    """
    Forth order derivative in 1D, assumes periodic boundary condition.
    https://en.wikipedia.org/wiki/Finite_difference_coefficient
    Args:
        dx (float): spatial discretization
        kernel_size (int): choices=[5, 7]
        device (PyTorch device): active device
    """
    def __init__(self, dx, kernel_size=5, device='cpu'):
        super().__init__()
        self.dx = dx

        # Second order accurate
        WEIGHT_5 = torch.FloatTensor([[[1, -4, 6, -4, 1]]]).to(device)

        # Forth order accurate
        WEIGHT_7 = torch.FloatTensor([[[-1, 12, -39, 56, -39, 12, -1]]]).to(device) / 6.

        if kernel_size == 5:
            self.padding = _pair(2)
            self.weight = WEIGHT_5
        
        elif kernel_size == 7:
            self.padding = _pair(3)
            self.weight = WEIGHT_7

    def __call__(self, u):
        """
        Args:
            u (Tensor): [B, C, H]
        Returns:
            div_u: [B, C, H]

        """
        u_shape = u.shape
        u = u.view(-1, 1, *u_shape[-1:])
        u = F.conv1d(F.pad(u, self.padding, mode='circular'), 
            self.weight, stride=1, padding=0, bias=None) / (self.dx**4)
        return u.view(u_shape)

class LaplaceFilter1D(object):
    """
    Second order derivative in 1D, assumes periodic boundary condition.
    https://en.wikipedia.org/wiki/Finite_difference_coefficient
    Args:
        dx (float): spatial discretization, assumes dx = dy
        kernel_size (int): choices=[3, 5]
        device (PyTorch device): active device
    """
    def __init__(self, dx, kernel_size=3, device='cpu'):
        super().__init__()
        self.dx = dx
        # Second order accurate
        WEIGHT_3 = torch.FloatTensor([[[1, -2, 1]]]).to(device)

        # Forth order accurate
        WEIGHT_5 = torch.FloatTensor([[[-1, 16, -30, 16, -1]]]).to(device) / 12.

        if kernel_size == 3:
            self.padding = _pair(1)
            self.weight = WEIGHT_3

        elif kernel_size == 5:
            self.padding = _pair(2)
            self.weight = WEIGHT_5

    def __call__(self, u):
        """
        Args:
            u (torch.Tensor): [B, C, H]
        Returns:
            div_u (torch.Tensor): [B, C, H]
        """
        u_shape = u.shape
        u = u.view(-1, 1, *u_shape[-1:])
        u = F.conv1d(F.pad(u, self.padding, mode='circular'), 
            self.weight, stride=1, padding=0, bias=None) / (self.dx**2)
        return u.view(u_shape)


class FluxWENOFilter1D(object):
    """
    3rd order WENO flux limiter
    https://math.la.asu.edu/~gardner/weno.pdf
    https://link.springer.com/chapter/10.1007/978-3-662-03882-6_5
    Args:
        dx (float): spatial discretization
        kernel_size (int): choices=[3, 5]
        device (PyTorch device): active device
    """
    def __init__(self, dx, limiter=1, device='cpu'):
        self.dx = dx

        
        # Gradient ratio convolutions
        self.weight_flux1 = torch.FloatTensor([[[-0.5, 1.5, 0]]]).to(device) / 1.
        self.weight_flux2 = torch.FloatTensor([[[0, 0.5, 0.5]]]).to(device) / 1.

        # Gradient ratio convolutions
        self.weight_beta1 = torch.FloatTensor([[[-1, 1, 0]]]).to(device) / 1.
        self.weight_beta2 = torch.FloatTensor([[[0, -1, 1]]]).to(device) / 1.

        self.weight = torch.FloatTensor([[[-1, 0, 1]]]).to(device) / 2.

        self.padding = _pair(1)       

    def __call__(self, u):
        """
        Args:
            u (torch.Tensor): (B, C, H)
        Returns:
            grad_u: (B, C, H)
        """
        u_shape = u.shape
        u = u.view(-1, 1, *u_shape[-1:])

        flux = u**2/2.0
        
        edge_flux = self.calcEdgeFlux(flux)
        edge_flux_r = self.calcEdgeFlux(torch.flip(flux, dims=[-1]))
        edge_flux_r = torch.flip(edge_flux_r, dims=[-1])

        flux_grad = (edge_flux[:,:,1:] - edge_flux[:,:,:-1])/self.dx
        flux_grad_r = (edge_flux_r[:,:,1:] - edge_flux_r[:,:,:-1])/self.dx
        
        # with torch.no_grad():
        #     grad = F.conv1d(F.pad(u, (1,1), mode='circular'), 
        #         self.weight, stride=1, padding=0, bias=None) / (self.dx)

        flux_grad = torch.where(u < 0, flux_grad, flux_grad_r)

        return flux_grad.view(u_shape)

    def calcEdgeFlux(self, flux):

        flux_edge1 = F.conv1d(F.pad(flux, (2,1), mode='circular'), self.weight_flux1, 
                        stride=1, padding=0, bias=None)

        flux_edge2 = F.conv1d(F.pad(flux, (1,2), mode='circular'), self.weight_flux2, 
                        stride=1, padding=0, bias=None)

        beta1 = torch.pow(F.conv1d(F.pad(flux, (2,1), mode='circular'), self.weight_beta1, 
                        stride=1, padding=0, bias=None), 2)
        
        beta2 = torch.pow(F.conv1d(F.pad(flux, (1,2), mode='circular'), self.weight_beta2, 
                        stride=1, padding=0, bias=None), 2)
        
        eps = 1e-6
        w1 = 1./(3*(eps + beta1)**2)
        w2 = 2./(3*(eps + beta2)**2)

        w = torch.stack([w1, w2], dim = 0)

        w = w / torch.sum(w, dim=0)

        edge_flux = w[0]*flux_edge1 + w[1]*flux_edge2

        return edge_flux

class GradFilter1D(object):
    """
    1st-order gradient in 1D. Assumes periodic boundary condition.
    https://en.wikipedia.org/wiki/Finite_difference_coefficient
    Args:
        dx (float): spatial discretization
        kernel_size (int): choices=[3, 5]
        device (PyTorch device): active device
    """
    def __init__(self, dx, kernel_size=3, device='cpu'):
        self.dx = dx
        self.kernel_size = kernel_size

        # Euler (first order accurate)
        WEIGHT_2 = torch.FloatTensor([[[1, -4, 3, 0, 0]]]).to(device) / 2.
        WEIGHT_2 = torch.FloatTensor([[[-1, 1, 0]]]).to(device)

        # Central difference (second order accurate)
        WEIGHT_3 = torch.FloatTensor([[[-1, 0, 1]]]).to(device) / 2.

        # Forth order accurate
        WEIGHT_5 = torch.FloatTensor([[[1, -8, 0, 8, -1]]]).to(device) / 12.

        # Fifth order accurate
        WEIGHT_7 = torch.FloatTensor([[[-1, 9, -45, 0, 45, -9, 1]]]).to(device) / 60.

        if kernel_size == 2:
            self.weight = WEIGHT_2
            self.padding = _pair(1)
        elif kernel_size == 3:
            self.weight = WEIGHT_3
            self.padding = _pair(1)
        elif kernel_size == 5:
            self.weight = WEIGHT_5
            self.padding = _pair(2)  
        elif kernel_size == 7:
            self.weight = WEIGHT_7
            self.padding = _pair(3)      

    def __call__(self, u):
        """
        Args:
            u (torch.Tensor): [B, C, H]
        Returns:
            grad_u: [B, C, H]
        """
        if(self.kernel_size == 2):
            return self.conditionalUpwind(u)

        u_shape = u.shape
        u = u.view(-1, 1, *u_shape[-1:])
        u = F.conv1d(F.pad(u, self.padding, mode='circular'), 
            self.weight, stride=1, padding=0, bias=None) / (self.dx)

        return u.view(u_shape)

    def conditionalUpwind(self, u):
        """
        Upwind scheme:
        https://en.wikipedia.org/wiki/Upwind_scheme
        Args:
            u (torch.Tensor): [B, C, H]
        Returns:
            grad_u: [B, C, H]
        """
        u_shape = u.shape
        u = u.view(-1, 1, *u_shape[-1:])

        u1 = F.conv1d(F.pad(u, self.padding, mode='circular'), 
            self.weight, stride=1, padding=0, bias=None) / (self.dx)

        u2 = F.conv1d(F.pad(u, self.padding, mode='circular'), 
            -torch.flip(self.weight, dims=[-1]), stride=1, padding=0, bias=None) / (self.dx)

        u = torch.where(u > 0, u1, u2)

        return u2.view(u_shape)

class BurgerIntegrate(object):
    '''
    Performs time-integration of the viscous 1D Burgers' equation
    Args:
        dx (float): spatial discretization
        nu (float): hyper-viscosity
        grad_kernels (list): list of kernel sizes for first, second and forth order gradients
        device (PyTorch device): active device
    '''
    def __init__(self, dx, nu=1.0, grad_kernels=[3, 3], device='cpu'):
        
        self.nu = nu

        # Create gradients
        self.grad1 = GradFilter1D(dx, kernel_size=grad_kernels[0], device=device)
        self.grad2 = LaplaceFilter1D(dx, kernel_size=grad_kernels[1], device=device)
        self.fluxgrad = FluxWENOFilter1D(dx, limiter=1, device=device)

    def backwardEuler(self, uPred, uPred0, dt):
        """
        Time integration of the 1D Burgers' system using implicit euler method
        Args:
            uPred (torch.Tensor): predicted quantity at t+dt
            uPred0 (torch.Tensor): prior timestep/input at t
            dt (float): delta t
        Returns:
            ustar (torch.Tensor): u integrated to time-step t+dt
        """
        # Calculate residual
        # Flux derivative
        fluxdx = 0.5*self.grad1(torch.pow(uPred, 2))
        # Second order derivative
        udx2 = self.grad2(uPred)
        # Regular implicit finite difference
        ustar = uPred0 + dt*(-0.5*fluxdx + self.nu*(udx2))

        return ustar

    def crankNicolson(self, uPred, uPred0, dt):
        """
        Time integration of the 1D Burgers' system using crank-nicolson
        Args:
            uPred (torch.Tensor): predicted quantity at t+dt
            uPred0 (torch.Tensor): prior timestep/input at t
            dt (float): delta t
        Returns:
            ustar (torch.Tensor): u integrated to time-step t+dt
        """
        fluxdx = self.grad1(uPred)
        udx2 = self.grad2(uPred)

        fluxdx0 = self.grad1(uPred0)
        udx20 = self.grad2(uPred0)

        ustar = uPred0  - 0.5*dt*((uPred*fluxdx + uPred0*fluxdx0) - self.nu*(udx2 + udx20))

        return ustar

    def crankNicolsonLimiter(self, uPred, uPred0, dt):
        """
        Time integration of the 1D Burgers' system using crank-nicolson with a flux-limiter
        Args:
            uPred (torch.Tensor): predicted quantity at t+dt
            uPred0 (torch.Tensor): prior timestep/input at t
            dt (float): delta t
        Returns:
            ustar (torch.Tensor): u integrated to time-step t+dt
        """
        fluxdx = self.fluxgrad(uPred)
        udx2 = self.grad2(uPred)

        fluxdx0 = self.fluxgrad(uPred0)
        udx20 = self.grad2(uPred0)

        ustar = uPred0  - 0.5*dt*((fluxdx + fluxdx0) - self.nu*(udx2 + udx20))

        return ustar


if __name__ == '__main__':

    # Gradient tests
    dx = 2*np.pi/101.
    x = torch.FloatTensor(np.linspace(0, 2*np.pi, 101)[:-1])

    y = torch.zeros(100) - 1
    y[:50] = 1

    fluxgrad = GradFilter1D(dx, kernel_size=3, device='cpu')

    out = fluxgrad(y.unsqueeze(0).unsqueeze(0).unsqueeze(0))

    plt.plot(x.numpy(), y.numpy())
    plt.plot(x.numpy(), out.squeeze().numpy())
    plt.show()