import copy, os
import numpy as np

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.distributions import Gamma

class SwagNN(torch.nn.Module):
    '''
    Stochastic Weighted Average Guassian Implementation
    Reference: https://arxiv.org/abs/1902.02476
    Original Code: https://github.com/wjmaddox/swa_gaussian/

    This implementation is built ontop of the Bayesian neural network class
    and is designed to control the samples of the parameters in the BayesNN
    Args:
        args: global arguements
        base (torch.nn.Module): Base PyTorch model with learnable parameters
        full_cov (boolean): Use full covariance matrix or just diagonal
        max_models (int): Maximum number of base models to store to compute 
            the covariance matrix. Not needed if just usting diagonal
        eps (float): Min variance for numerical stability
    '''
    def __init__(self, args, base, full_cov=True, max_models=1, eps=1e-30):
        super(SwagNN, self).__init__()

        self.device = args.device
        self.base = base

        self.full_cov = full_cov
        self.max_models = max_models
        self.eps = eps

        # Build swag parameters
        self.buildSwagParameters(base)

    def buildSwagParameters(self, baseModule):
        '''
        Builds Swag parameters used to keep track of sampled model params
        Args:
            baseModule (torch.nn.Module): Base PyTorch model with learnable parameters
                that is used to structure the tracked params by swag
        '''
        # Only track learnable parameters
        # Parameters such as running means/variance in batch norms are not sampled
        for (name, data) in list(baseModule.named_parameters()):
            # Ignore un-named parameters
            if name is None:
                continue
            # Need to replace periods in name, these are not allows in buffer names
            # https://github.com/pytorch/vision/pull/474
            name = name.replace('.','_')

            # Use register buffer, basically a parameter but wont track gradients
            self.register_buffer('%s_mean' % name, data.new(data.size()).zero_())
            self.register_buffer('%s_sq_mean' % name, data.new(data.size()).zero_())

            if self.full_cov is True:
                self.register_buffer( '%s_cov_mat_sqrt' % name, data.new_empty((self.max_models, data.numel())).zero_() )

        # Number of models samples
        self.register_buffer('n_models', torch.zeros([1], dtype=torch.long))

    def forward(self, *args, **kwargs):
        return self.base(*args, **kwargs)

    def calc_neg_log_joint(self, *args, **kwargs):
        return self.base.calc_neg_log_joint(*args, **kwargs)

    def swag_state_dict(self):
        '''
        Returns dictionary of swag parameters, this excludes parameters in the base model
        https://discuss.pytorch.org/t/how-to-get-all-registerd-buffer-by-self-register-buffer/18883
        '''
        return dict(filter(lambda v: (v[1].requires_grad==False and "base" not in v[0]), \
            self.state_dict().items()))

    def sample(self, module=None, scale=1.0, diagCov=False, copy_model=True, seed=None):
        '''
        Samples model parameters from the approx. posterior and sets them on the specified model.
        If the provided module is None and copy is False, this will update the parameters of the
        base model.
        Args:
            module (torch.nn.Module): module that must be the same class as the base
            scale (float): custom scaling of the posterior, default is 1.0
            diagCov (boolean): If to use a diagonal covariance for sampling or full rank
            copy (boolean): If the provided model should be deep copied
            seed (int): Randome seed for sampling
        Returns
            module (torch.nn.Module): neural network model with sampled parameters
        '''
        if(module is None):
            module = self.base
        assert type(module) is type(self.base), 'Provided module is not the same as base module!'

        if seed is not None:
            torch.manual_seed(seed)

        if copy_model:
            module0 = copy.deepcopy(module)
        else:
            module0 = module

        # Here sample each parameter individually since the total number
        # of parameters in a model could be in the millions. Even for the
        # the non-diagonal version, we can due this thanks to the low-rank approx.
        # If youre stupid like me:
        # https://math.stackexchange.com/questions/2848517/sampling-multivariate-normal-with-low-rank-covariance
        for (name0, param) in list(module0.named_parameters()):

            # Replace periods in name
            name = name0.replace('.','_')
            # Get swag params
            mean = self.__getattr__('%s_mean' % name)
            sq_mean = self.__getattr__('%s_sq_mean' % name)
            
            var = torch.clamp(sq_mean - mean ** 2, min=self.eps)
            # if(name == 'model_log_beta'):
            #     print(mean)
            #     print(sq_mean)
            #     print(var)

            # Sample from  unit normal and scale it by the std
            scaled_diag_sample = scale * torch.sqrt(var) * torch.randn_like(mean)
            # if(name == 'model_features_DecBlock1_denselayer6_conv1_weight'):
                # print('var:',torch.sqrt(var))
                # print('scaled diag',scaled_diag_sample)
                # print('mean',mean)
            

            # If we wish to use the low-order approx Sigma ~ Sigma_diag + DD^T
            if diagCov is False:
                # Subtract SWAG mean from parameters
                cov_mat_sqrt = self.__getattr__('%s_cov_mat_sqrt' % name)
                cov_mat_sqrt = torch.where(cov_mat_sqrt == 0, cov_mat_sqrt, cov_mat_sqrt - mean.view(-1).unsqueeze(0))
                # Sample unit normal
                rand = cov_mat_sqrt.new_empty((cov_mat_sqrt.size(0), 1)).normal_()
                # Scale by sqrt(D)
                cov_sample = (scale/((self.max_models - 1) ** 0.5)) * cov_mat_sqrt.t().matmul(rand).view_as(mean)
                # Add mean and variance
                w = mean + np.sqrt(0.5)*(scaled_diag_sample + cov_sample)
            else:
                # Add mean
                w = mean + scaled_diag_sample
            # Update the parameter
            param.data = w

        return module0

    def collect(self, module=None):
        '''
        Collects parameters from the provided module and updates swag statisitcs
        Args:
            module (torch.nn.Module): module that must be the same class as the base
                module that we wish to sample the parameters from. If none is provided
                the base module will be used.
        '''
        if(module is None):
            module = self.base
        assert type(module) is type(self.base), 'Provided module is not the same as base module!'

        for idx, (name, param) in enumerate(list(module.named_parameters())):
            # Replace periods in name
            name = name.replace('.','_')

            # Get current moments
            mean = self.__getattr__('%s_mean' % name)
            sq_mean = self.__getattr__('%s_sq_mean' % name)
            n = self.n_models.item()

            # Update the mean
            mean = mean * n / (n + 1.0) + param.data / (n + 1.0)
            
            # Update the second moment
            sq_mean = sq_mean * n / (n + 1.0) + param.data ** 2 / (n + 1.0)

            # If full covatiance matrix
            if self.full_cov:
                cov_mat_sqrt = self.__getattr__('%s_cov_mat_sqrt' % name)
                # We store deviation from current mean
                # See section 3.3 of reference
                dev = (param.data).view(-1,1)
                cov_mat_sqrt = torch.cat((cov_mat_sqrt[-(self.max_models-1):], dev.view(-1,1).t()),dim=0)

                self.__setattr__('%s_cov_mat_sqrt' % name, cov_mat_sqrt)

            # Update the buffers
            self.__setattr__('%s_mean' % name, mean)
            self.__setattr__('%s_sq_mean' % name, sq_mean)

        # Update number of samples
        self.n_models.add_(1)


    def saveModel(self, epoch, optimizer, scheduler, file_dir):
        '''
        Save neural network
        '''
        # If director does not exist create it
        if not os.path.exists(file_dir):
            os.makedirs(file_dir)

        print('[SWAG] Epoch {}, Saving SWAG model!'.format(epoch))
        # Create state dict of both the model and optimizer
        state = {'epoch': epoch, 'state_dict': self.state_dict(),
             'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict()}
        torch.save(state, file_dir+'/torchSwagModel_epoch{:d}.pth'.format(epoch))

    def loadModel(self, epoch, optimizer=None, scheduler=None, file_dir="."):
        '''
        Loads pre-trained network from file
        '''
        try:
            file_name = file_dir+'/torchSwagModel_epoch{:d}.pth'.format(epoch)
            param_dict = torch.load(file_name, map_location=lambda storage, loc: storage)
            print('[SWAG] Found model at epoch: {:d}'.format(param_dict['epoch']))
        except FileNotFoundError:
            print('[SWAG] Error: Could not find PyTorch network')
            return
        # Load swag model
        self.load_state_dict(param_dict['state_dict'])
        # Load optimizer/scheduler
        if(not optimizer is None):
            optimizer.load_state_dict(param_dict['optimizer'])
            scheduler.load_state_dict(param_dict['scheduler'])
        print('[SWAG] Pre-trained swag model loaded!')
        print('[SWAG] Collected {:d} models'.format( self.n_models.item()))

        return optimizer, scheduler