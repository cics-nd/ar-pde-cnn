import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair, _quadruple

class FourthOrderFilter1D(object):
    """
    Forth order derivative in 1D, assumes periodic boundary condition.
    https://en.wikipedia.org/wiki/Finite_difference_coefficient
    Args:
        dx (float): spatial discretization
        kernel_size (int): choices=[5, 7]
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
            u (Tensor): (B, C, H)
        Returns:
            div_u: (B, C, H)

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
            u (Tensor): (B, C, H)
        Returns:
            div_u: (B, C, H)
        """
        u_shape = u.shape
        u = u.view(-1, 1, *u_shape[-1:])
        u = F.conv1d(F.pad(u, self.padding, mode='circular'), 
            self.weight, stride=1, padding=0, bias=None) / (self.dx**2)
        return u.view(u_shape)


class GradFilter1D(object):
    """
    1st-order gradient in 1D. Assumes periodic boundary condition.
    https://en.wikipedia.org/wiki/Finite_difference_coefficient
    Args:
        dx (float): spatial discretization
        kernel_size (int): choices=[3, 5]
    """
    def __init__(self, dx, kernel_size=3, device='cpu'):
        self.dx = dx
        # Central difference (second order accurate)
        WEIGHT_3 = torch.FloatTensor([[[-1, 0, 1]]]).to(device) / 2.

        # Forth order accurate
        WEIGHT_5 = torch.FloatTensor([[[1, -8, 0, 8, -1]]]).to(device) / 12.

        if kernel_size == 3:
            self.weight = WEIGHT_3
            self.padding = _pair(1)
        elif kernel_size == 5:
            self.weight = WEIGHT_5
            self.padding = _pair(2)        

    def __call__(self, u):
        """
        Args:
            u (Tensor): (B, C, H)
        Returns:
            grad_u: (B, C, H)
        """
        u_shape = u.shape
        u = u.view(-1, 1, *u_shape[-1:])
        u = F.conv1d(F.pad(u, self.padding, mode='circular'), 
            self.weight, stride=1, padding=0, bias=None) / (self.dx)
        return u.view(u_shape)

class KSIntegrate(object):
    '''
    Performs time-integration of the Kuramoto-Sivashinsky equation
    Args:
        dx (float): spatial discretization
        nu (float): hyper-viscosity
        grad_kernels (list): list of kernel sizes for first, second and forth order gradients
        device (PyTorch device): active device
    '''
    def __init__(self, dx, nu=1.0, grad_kernels=[3, 3, 5], device='cpu'):
        
        self.nu = nu

        # Create gradients
        self.grad1 = GradFilter1D(dx, kernel_size=grad_kernels[0], device=device)
        self.grad2 = LaplaceFilter1D(dx, kernel_size=grad_kernels[1], device=device)
        self.grad4 = FourthOrderFilter1D(dx, kernel_size=grad_kernels[2], device=device)

    def backwardEuler(self, uPred, uPred0, dt):
        """
        Time integration of the 1D K-S system using implicit euler method
        Args:
            uPred (torch.Tensor): predicted quantity at t+dt
            uPred0 (torch.Tensor): prior timestep/input at t
            dt (float): delta t
        """
        fluxdx = self.grad1(uPred**2)
        udx2 = self.grad2(uPred)
        udx4 = self.grad4(uPred)

        ustar = uPred0 - dt*(0.5*fluxdx + udx2 + self.nu *udx4)

        return ustar

    def crankNicolson(self, uPred, uPred0, dt):
        """
        Time integration of the 1D K-S system using crank-nicolson
        Args:
            uPred (torch.Tensor): predicted quantity at t+dt
            uPred0 (torch.Tensor): prior timestep/input at t
            dt (float): delta t
        """
        fluxdx = self.grad1(uPred**2)
        udx2 = self.grad2(uPred)
        udx4 = self.grad4(uPred)

        fluxdx0 = self.grad1(uPred0**2)
        udx20 = self.grad2(uPred0)
        udx40 = self.grad4(uPred0)

        ustar = uPred0 - 0.5*dt*((0.5*fluxdx + udx2 + self.nu *udx4) + (0.5*fluxdx0 + udx20 + self.nu *udx40))

        return ustar