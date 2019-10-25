import os
import numpy as np

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.distributions import Gamma

class BayesNN(nn.Module):
    '''
    Bayesian neural network, when used alone one will arrive at the MAP estimate
    Which can be interpreted as regularized L2 loss.
    Args:
        args (argparse): object with programs arguements
        model (torch.nn.Module): Dense encoder-decoder model
    '''
    def __init__(self, args, model):
        super(BayesNN, self).__init__()

        # Time-step size of nn
        self.dt = args.dt
        # PyTorch device
        self.device = args.device

        # Predict expected value of epistemic noise percision
        # Based on truncation error prediction of the numerical integrator
        # Prior is weak and beta is learnable so value doesnt matter signifcantly
        c0 = 0.2
        k0 = 3.0 # Predicted order of convergence of time
        beta_mean = 1.0/(c0*args.dt**(k0))

        # Hyper parameters of the additive output wise noise
        self.beta_prior_shape = 10.
        self.beta_prior_rate = self.beta_prior_shape/beta_mean

        # Student's t-distribution: w ~ St(w | mu=0, lambda=shape/rate, nu=2*shape)
        # See PRML by Bishop Page 103
        self.w_prior_shape = 0.5
        self.w_prior_rate = 10

        # Dense ED model
        self.model = model

        # Now set up parameters for the tuncation error noise hyper-prior
        beta0 = np.log(np.random.gamma(self.beta_prior_shape,
                        1. / self.beta_prior_rate, size=(1)))

        self.model.log_beta = Parameter(torch.Tensor(beta0).to(self.device))

        print('[BayesNN]: Expected Noise: {:.03f}'.format(beta_mean))
        print('[BayesNN]: Total number of parameters: {}'.format(self._num_parameters()))

    def _num_parameters(self):
        count = 0
        for name, param in self.named_parameters():
            # print(name)
            count += param.numel()
        return count

    def beta(self):
        '''
        Get output noise parameter
        '''
        return self.model.log_beta.exp().item()

    def forward(self, input):
        """
        Forward of underlying Dense Encoder-Decoder
        Args:
            input: N x iC x iH 
        Return:
            output: N x oC x oH 
        """
        output = self.model(input)

        return output

    def predict(self, x_test):
        """
        Predictive mean and variance at x_test.
        TODO: Couple with SWAG
        Args:
            x_test (Tensor): [N, *], test input
        """
        # S x N x oC x oH x oW
        y = self.forward(x_test)
        y_pred_mean = y.mean(0)
        # compute predictive variance per pixel
        # N x oC x oH x oW
        EyyT = (y ** 2).mean(0)
        EyEyT = y_pred_mean ** 2
        beta_inv = (-self.c0)*self.dt**(self.k0)
        y_pred_var = beta_inv.mean() + EyyT - EyEyT

        return y_pred_mean, y_pred_var

    def calc_neg_log_joint(self, output, target, ntrain):
        """
        Negative log joint probability or unnormalized posterior which consists of
        the log likelihood and the Gamma prior of the output additive noise.
        No prior (non-informative) is placed on the weights, however regulazation
        and momentum in ADAM places an implicit complex prior on the weights.
        Args:
            output (Tensor): B x oC x oH x oH
            target (Tensor): B x oC x oH x oH
            ntrain (int): total number of training data, mini-batch is used to
                evaluate the log joint prob
        Returns:
            Log joint probability (zero-dim tensor)
        """
        # log Normal(target | output, 1 / beta * I)
        log_likelihood = ntrain / output.size(0) * (- 0.5 * self.model.log_beta.exp()
                            * (target - output).pow(2).sum()
                            + 0.5 * target.numel() * self.model.log_beta)

        # log prob of prior of weights, i.e. log prob of studentT
        log_prob_prior_w = torch.tensor(0.).to(self.device)
        for param in self.model.features.parameters():
            log_prob_prior_w += \
                torch.log1p(0.5 / self.w_prior_rate * param.pow(2)).sum()
        log_prob_prior_w *= -(self.w_prior_shape + 0.5)

        # Log Gamma Output-wise noise prior
        prior_log_beta = ((self.beta_prior_shape - 1.0) * self.model.log_beta \
                    - self.model.log_beta.exp() * self.beta_prior_rate)

        # print(log_likelihood, prior_log_beta)
        return - log_likelihood - log_prob_prior_w - prior_log_beta


    def saveNetwork(self, epoch, file_dir='./networks'):
        '''
        Save neural network
        '''
        # If director does not exist create it
        if not os.path.exists(file_dir):
            os.makedirs(file_dir)

        print('Epoch {}, Saving network!'.format(epoch))
        torch.save(self.state_dict(), file_dir+'/torchNet_epoch{:d}.pth'.format(epoch))

    def loadNetwork(self, epoch, file_dir='./networks'):
        '''
        Loads pre-trained network from file
        '''
        try:
            print('[BayesNN]: Attempting to load network from epoch {:d}'.format(epoch))
            file_name = file_dir+'/torchNet_epoch{:d}.pth'.format(epoch)
            param_dict = torch.load(file_name, map_location=lambda storage, loc: storage)
        except FileNotFoundError:
            print('Error: Could not find PyTorch network')
            return

        self.load_state_dict(param_dict)
        print('[BayesNN]: Pre-trained network loaded!')
