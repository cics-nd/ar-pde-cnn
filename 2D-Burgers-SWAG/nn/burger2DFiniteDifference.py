import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair, _quadruple

class LapLaceFilter2d(object):
    """
    Smoothed Laplacian 2D, assumes periodic boundary condition.
    https://en.wikipedia.org/wiki/Finite_difference_coefficient
    Args:
        dx (float): spatial discretization, assumes dx = dy
        kernel_size (int): choices=[3, 5]
        device (PyTorch device): active device
    """
    def __init__(self, dx, kernel_size=3, device='cpu'):
        super().__init__()
        self.dx = dx
        # no smoothing
        WEIGHT_3x3 = torch.FloatTensor([[[[0, 1, 0],
                                          [1, -4, 1],
                                          [0, 1, 0]]]]).to(device)
        # smoothed
        WEIGHT_3x3 = torch.FloatTensor([[[[1, 2, 1],
                                          [-2, -4, -2],
                                          [1, 2, 1]]]]).to(device) / 4.

        WEIGHT_3x3 = WEIGHT_3x3 + torch.transpose(WEIGHT_3x3, -2, -1)

        print(WEIGHT_3x3)

        WEIGHT_5x5 = torch.FloatTensor([[[[0, 0, -1, 0, 0],
                                          [0, 0, 16, -0, 0],
                                          [-1, 16, -60, 16, -1],
                                          [0, 0, 16, 0, 0],
                                          [0, 0, -1, 0, 0]]]]).to(device) / 12.
        if kernel_size == 3:
            self.padding = _quadruple(1)
            self.weight = WEIGHT_3x3
        elif kernel_size == 5:
            self.padding = _quadruple(2)
            self.weight = WEIGHT_5x5

    def __call__(self, u):
        """
        Args:
            u (torch.Tensor): [B, C, H, W]
        Returns:
            div_u(torch.Tensor): [B, C, H, W]
        """
        u_shape = u.shape
        u = u.view(-1, 1, *u_shape[-2:])
        u = F.conv2d(F.pad(u, self.padding, mode='circular'), 
            self.weight, stride=1, padding=0, bias=None) / (self.dx**2)
        return u.view(u_shape)


class SobelFilter2d(object):
    """
    Sobel filter to estimate 1st-order gradient in horizontal & vertical 
    directions. Assumes periodic boundary condition.
    Args:
        dx (float): spatial discretization, assumes dx = dy
        kernel_size (int): choices=[3, 5]
        device (PyTorch device): active device
    """
    def __init__(self, dx, kernel_size=3, device='cpu'):
        self.dx = dx
        # smoothed central finite diff
        WEIGHT_H_3x3 = torch.FloatTensor([[[[-1, 0, 1],
                                            [-2, 0, 2],
                                            [-1, 0, 1]]]]).to(device) / 8.

        # larger kernel size tends to smooth things out
        WEIGHT_H_5x5 = torch.FloatTensor([[[[1, -8, 0, 8, -1],
                                            [2, -16, 0, 16, -2],
                                            [3, -24, 0, 24, -3],
                                            [2, -16, 0, 16, -2],
                                            [1, -8, 0, 8, -1]]]]).to(device) / (9*12.)
        if kernel_size == 3:
            self.weight_h = WEIGHT_H_3x3
            self.weight_v = WEIGHT_H_3x3.transpose(-1, -2)
            self.weight = torch.cat((self.weight_h, self.weight_v), 0)
            self.padding = _quadruple(1)
        elif kernel_size == 5:
            self.weight_h = WEIGHT_H_5x5
            self.weight_v = WEIGHT_H_5x5.transpose(-1, -2)
            self.padding = _quadruple(2)        

    def __call__(self, u):
        """
        Compute both hor and ver grads
        Args:
            u (torch.Tensor): (B, C, H, W)
        Returns:
            grad_u: (B, C, 2, H, W), grad_u[:, :, 0] --> grad_h
                                     grad_u[:, :, 1] --> grad_v
        """
        # (B*C, 1, H, W)
        u_shape = u.shape
        u = u.view(-1, 1, *u_shape[-2:])
        u = F.conv2d(F.pad(u, self.padding, mode='circular'), self.weight, 
                        stride=1, padding=0, bias=None) / (self.dx)
        return u.view(*u_shape[:2], *u.shape[-3:])

    def grad_h(self, u):
        """
        Get image gradient along horizontal direction, or x axis.
        Perioid padding before conv.
        Args:
            u (torch.Tensor): [B, C, H, W]
        Returns:
            ux (torch.Tensor): [B, C, H, W] calculated gradient
        """
        ux = F.conv2d(F.pad(u, self.padding, mode='circular'), self.weight_h, 
                        stride=1, padding=0, bias=None) / (self.dx)
        return ux
    
    def grad_v(self, u):
        """
        Get image gradient along vertical direction, or y axis.
        Perioid padding before conv.
        Args:
            u (torch.Tensor): [B, C, H, W]
        Returns:
            uy (torch.Tensor): [B, C, H, W] calculated gradient
        """
        uy = F.conv2d(F.pad(u, self.padding, mode='circular'), self.weight_v, 
                        stride=1, padding=0, bias=None) / (self.dx)
        return uy

class Burger2DIntegrate(object):
    '''
    Performs time-integration of the 2D Burger equation
    Args:
        dx (float): spatial discretization
        nu (float): hyper-viscosity
        grad_kernels (list): list of kernel sizes for first, second and forth order gradients
        device (PyTorch device): active device
    '''
    def __init__(self, dx, nu=1.0, grad_kernels=[3, 3], device='cpu'):
        
        self.nu = nu

        # Create gradients
        self.grad1 = SobelFilter2d(dx, kernel_size=grad_kernels[0], device=device)
        self.grad2 = LapLaceFilter2d(dx, kernel_size=grad_kernels[1], device=device)

    def backwardEuler(self, uPred, uPred0, dt):
        """
        Time integration of the 2D Burger system using implicit euler method
        Args:
            uPred (torch.Tensor): predicted quantity at t+dt
            uPred0 (torch.Tensor): prior timestep/input at t
            dt (float): delta t
        Returns:
            ustar (torch.Tensor): u integrated to time-step t+dt
        """
        grad_ux = self.grad1.grad_h(0.5*uPred[:,:1,:,:]**2)
        grad_uy = self.grad1.grad_v(uPred[:,:1,:,:])

        grad_vx = self.grad1.grad_h(uPred[:,1:,:,:])
        grad_vy = self.grad1.grad_v(0.5*uPred[:,1:,:,:]**2)

        div_u = self.nu * self.grad2(uPred[:,:1,:,:])
        div_v = self.nu * self.grad2(uPred[:,1:,:,:])

        burger_rhs_u = -grad_ux - uPred[:,1:,:,:]*grad_uy + div_u
        burger_rhs_v = -uPred[:,:1,:,:]*grad_vx - grad_vy + div_v

        ustar_u = uPred0[:,:1,:,:] + dt * burger_rhs_u
        ustar_v = uPred0[:,1:,:,:] + dt * burger_rhs_v

        return torch.cat([ustar_u, ustar_v], dim=1)

    def crankNicolson(self, uPred, uPred0, dt):
        """
        Time integration of the 2D Burger system using crank-nicolson
        Args:
            uPred (torch.Tensor): predicted quantity at t+dt
            uPred0 (torch.Tensor): prior timestep/input at t
            dt (float): delta t
        Returns:
            ustar (torch.Tensor): u integrated to time-step t+dt
        """
        grad_ux = self.grad1.grad_h(0.5*uPred[:,:1,:,:]**2)
        grad_uy = self.grad1.grad_v(uPred[:,:1,:,:])

        grad_vx = self.grad1.grad_h(uPred[:,1:,:,:])
        grad_vy = self.grad1.grad_v(0.5*uPred[:,1:,:,:]**2)

        div_u = self.nu * self.grad2(uPred[:,:1,:,:])
        div_v = self.nu * self.grad2(uPred[:,1:,:,:])

        grad_ux0 = self.grad1.grad_h(0.5*uPred0[:,:1,:,:]**2)
        grad_uy0 = self.grad1.grad_v(uPred0[:,:1,:,:])

        grad_vx0 = self.grad1.grad_h(uPred0[:,1:,:,:])
        grad_vy0 = self.grad1.grad_v(0.5*uPred0[:,1:,:,:]**2)

        div_u0 = self.nu * self.grad2(uPred0[:,:1,:,:])
        div_v0 = self.nu * self.grad2(uPred0[:,1:,:,:])
        
        burger_rhs_u = -grad_ux - uPred[:,1:,:,:]*grad_uy + div_u
        burger_rhs_v = -uPred[:,:1,:,:]*grad_vx - grad_vy + div_v
        burger_rhs_u0 = -grad_ux0 - uPred0[:,1:,:,:]*grad_uy0 + div_u0
        burger_rhs_v0 = -uPred0[:,:1,:,:]*grad_vx0 - grad_vy0 + div_v0

        ustar_u = uPred0[:,:1,:,:] + 0.5 * dt * (burger_rhs_u + burger_rhs_u0)
        ustar_v = uPred0[:,1:,:,:] + 0.5 * dt * (burger_rhs_v + burger_rhs_v0)

        return torch.cat([ustar_u, ustar_v], dim=1)