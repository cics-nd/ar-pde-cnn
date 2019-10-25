'''
2D Coupled Burgers' system model
===
Distributed by: Notre Dame CICS (MIT Liscense)
- Associated publication:
url: http://www.sciencedirect.com/science/article/pii/S0021999119307612
doi: https://doi.org/10.1016/j.jcp.2019.109056
github: https://github.com/cics-nd/ar-pde-cnn
===
'''
from args import Parser
from nn.denseEDcirc2d import DenseED
from nn.bayesNN import BayesNN
from nn.swag import SwagNN
from nn.burger2DFiniteDifference import Burger2DIntegrate
from utils.utils import mkdirs, toNumpy, toTuple
from utils.burgerLoader2D import BurgerLoader
from utils.post import plotPred, plotSamples
from torch.optim.lr_scheduler import ExponentialLR

import torch
import torch.nn.functional as F
import numpy as np
import os, time

def train(args, model, burgerInt, train_loader, optimizer, tsteps, tback, tstart, dt=0.1):
    '''
    Trains the model
    Args:
        args (argparse): object with programs arguements
        model (PyTorch model): SWAG DenseED model to be tested
        burgerInt (BurgerIntegrate): 1D Burger system time integrator
        train_loader (dataloader): dataloader with training cases (use createTrainingLoader)
        optimizer (Pytorch Optm): optimzer
        tsteps (np.array): [mb] number of timesteps to predict for each mini-batch
        tback (np.array): [mb] number of timesteps to forward predict before back prop
        tstart (np.array): [mb] time-step to start updating model (kept at 0 for now)
        dt (float): current time-step size of the model (used to progressively increase time-step size)
    Returns:
        loss_total (float): negative log joint posterior
        mse_total (float): mean square error between the prediction and time-integrator
    '''
    model.train()

    loss_total = 0
    mse_total = 0
    # Mini-batch loop
    for batch_idx, input in enumerate(train_loader):
        # input [b, 2, x, y]
        # Expand input to match model in channels
        dims = torch.ones(len(input.shape))
        dims[1] = args.nic 
        input = input.repeat(toTuple(toNumpy(dims).astype(int))).to(args.device)

        loss = 0
        
        # Loop for number of timesteps
        optimizer.zero_grad()
        for i in range(tsteps[batch_idx]):

            uPred = model(input[:,-2*args.nic:,:])
            
            if(i < tstart[batch_idx]):
                # Don't calculate residual, just predict forward
                input = input[:,-2*int(args.nic-1):,:].detach()
                input0 = uPred[:,0,:].unsqueeze(1).detach()
                input = torch.cat([input,  input0], dim=1)
            else:
                # Calculate loss
                # Start with implicit time integration
                ustar = burgerInt.crankNicolson(uPred, input[:,-2:,:], dt)
                # Calc. loss based on posterior of the model
                log_joint = model.calc_neg_log_joint(uPred, ustar, len(train_loader))
                loss = loss + log_joint
                
                loss_total = loss_total + loss.data.item()
                mse_total += F.mse_loss(uPred.detach(), ustar.detach()).item() # MSE for scheduler

                # Back-prop through two timesteps
                if((i+1)%tback[batch_idx] == 0):
                    loss.backward()
                    loss = 0

                    optimizer.step()
                    optimizer.zero_grad()
                    input = input[:,-2*int(args.nic-1):,:].detach()
                    input0 = uPred.detach()
                    input = torch.cat([input,  input0], dim=1)
                else:
                    input0 = uPred
                    input = torch.cat([input,  input0], dim=1)

        if(batch_idx % 10 == 1):
            print("Epoch {}, Mini-batch {}/{} ({}%) ".format(epoch, batch_idx, \
                len(train_loader), int(100*batch_idx/len(train_loader))))

    return loss_total/len(train_loader), mse_total/len(train_loader)


def test(args, model, test_loader, tstep=100, test_every=2):
    '''
    Tests the deterministic model
    Args:
        args (argparse): object with programs arguements
        model (PyTorch model): DenseED model to be tested
        test_loader (dataloader): dataloader with test cases (use createTestingLoader)
        tstep (int): number of timesteps to predict for
        test_every (int): Time-step interval to test (must match simulator), default = 2
    Returns:
        u_out (torch.Tensor): [d x (tstep+1)//test_every x 2 x nel x nel] predicted quantities
        u_target (torch.Tensor): [d x (tstep+1)//test_every x 2 x nel x nel] respective target values loaded from simulator
    '''
    model.eval()
    mb_size = int(len(test_loader.dataset)/len(test_loader))
    u_out = torch.zeros(len(test_loader.dataset), tstep//test_every+1, 2, args.nel, args.nel)
    u_target = torch.zeros(len(test_loader.dataset), tstep//test_every+1, 2, args.nel, args.nel)

    for bidx, (input0, uTarget0) in enumerate(test_loader):
        # Expand input to match model in channels
        dims = torch.ones(len(input0.shape))
        dims[1] = args.nic
        input = input0.repeat(toTuple(toNumpy(dims).astype(int))).to(args.device)

        u_out[bidx*mb_size:(bidx+1)*mb_size,0] = input0
        u_target[bidx*mb_size:(bidx+1)*mb_size] = uTarget0[:,:(tstep//test_every+1)].cpu()

        # Auto-regress
        for t_idx in range(tstep):
            uPred = model(input[:,-2*args.nic:,:,:])
            if((t_idx+1)%test_every == 0):
                u_out[bidx*mb_size:(bidx+1)*mb_size, (t_idx+1)//test_every,:,:,:] = uPred
            
            input = input[:,-2*int(args.nic-1):,:].detach()
            input0 = uPred.detach()
            input = torch.cat([input,  input0], dim=1)

    return u_out, u_target

def testSample(args, swag_nn, test_loader, tstep=100, n_samples=10, test_every=2):
    '''
    Tests the samples of the Bayesian SWAG model
    Args:
        args (argparse): object with programs arguements
        model (PyTorch model): DenseED model to be tested
        test_loader (dataloader): dataloader with test cases (use createTestingLoader)
        tstep (int): number of timesteps to predict for
        n_samples (int): number of model samples to draw
        test_every (int): Time-step interval to test (must match simulator), default = 2
    Returns:
        u_out (torch.Tensor): [d x nsamples x (tstep+1)//test_every x 2 x nel x nel] predicted quantities of each sample
        u_target (torch.Tensor): [d x (tstep+1)//test_every x 2 x nel x nel] respective target values loaded from simulator
    '''
    mb_size = int(len(test_loader.dataset)/len(test_loader))
    u_out = torch.zeros(len(test_loader.dataset), n_samples, (tstep)//test_every + 1, 2, args.nel, args.nel)
    u_target = torch.zeros(len(test_loader.dataset), (tstep)//test_every + 1, 2, args.nel, args.nel)
    
    for i in range(n_samples):
        print('Executing model sample {:d}'.format(i))
        model = swag_nn.sample(diagCov=True) # Use diagonal approx. only when training
        model.eval()

        for bidx, (input0, uTarget0) in enumerate(test_loader):
            # Expand input to match model in channels
            dims = torch.ones(len(input0.shape))
            dims[1] = args.nic
            input = input0.repeat(toTuple(toNumpy(dims).astype(int))).to(args.device)
            
            if(i == 0): # Save target data
                u_target[bidx*mb_size:(bidx+1)*mb_size] = uTarget0[:,:(tstep//test_every + 1)]
            u_out[bidx*mb_size:(bidx+1)*mb_size,i,0,:,:,:] = input0
            # Auto-regress
            for t_idx in range(tstep):
                uPred = model(input[:,-2*args.nic:,:])
                if((t_idx+1)%test_every == 0):
                    u_out[bidx*mb_size:(bidx+1)*mb_size, i, t_idx//test_every+1,:,:,:] = uPred

                input = input[:,-2*int(args.nic-1):,:].detach()
                input0 = uPred.detach()
                input = torch.cat([input,  input0], dim=1)

    return u_out, u_target

if __name__ == '__main__':

    # Parse arguements
    args = Parser().parse()
    use_cuda = "cpu"
    if(torch.cuda.is_available()):
        use_cuda = "cuda"
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Torch device:{}".format(args.device))
    
    # Domain settings, matches solver settings
    x0 = 0
    x1 = 1.0
    args.dx = (x1 - x0)/args.nel

    # Create training loader
    burgerLoader = BurgerLoader(dt=args.dt)
    training_loader = burgerLoader.createTrainingLoader(args.ntrain, args.nel, batch_size=args.batch_size)

    # Create training loader
    test_cases = np.array([400, 401]).astype(int)
    testing_loader = burgerLoader.createTestingLoader(args.data_dir, test_cases, simdt=0.005, batch_size=2)

    # Create DenseED model
    denseED = DenseED(in_channels=2*args.nic, out_channels=2*args.noc,
                        blocks=args.blocks,
                        growth_rate=args.growth_rate, 
                        init_features=args.init_features,
                        bn_size=args.bn_size,
                        drop_rate=args.drop_rate,
                        bottleneck=False,
                        out_activation=None).to(args.device)

    # Bayesian neural network
    bayes_nn = BayesNN(args, denseED)
    # Stochastic weighted averages
    swag_nn = SwagNN(args, bayes_nn, full_cov=True, max_models=args.swag_max)
    # Optimizer
    parameters = [{'params': [bayes_nn.model.log_beta], 'lr': args.lr_beta},
                    {'params': bayes_nn.model.features.parameters()}]
    optimizer = torch.optim.Adam(parameters, lr=args.lr, weight_decay=0.0)
    # Learning rate scheduler
    scheduler = ExponentialLR(optimizer, gamma=0.995)

    # If we are starting from a specific epoch, attempt to load a model
    if(args.epoch_start > 0):
        optimizer, scheduler = swag_nn.loadModel(args.epoch_start, optimizer, scheduler, file_dir=args.ckpt_dir)

    # Create Burger time integrator
    # Here we will use 2nd order finite differences for spacial derivatives
    burgerInt = Burger2DIntegrate(args.dx, nu=args.nu, grad_kernels=[3, 3], device=args.device)

    # Progressively increase the time step to help stabailize training
    dtStep = 25
    dtArr = np.linspace(np.log10(args.dt)-1, np.log10(args.dt), dtStep)
    dtArr = 10**(dtArr)

    # ========== Epoch loop ============
    print('>>> Training network, lets rock')
    for epoch in range(args.epoch_start+1, args.epochs + 1):

        if(epoch == args.swag_start):
            print('Starting to sample weights every {:d} epochs'.format(args.swag_freq))
            # Mannually set learning rate to swag sampling rate
            parameters = [{'params': [bayes_nn.model.log_beta], 'lr': args.swag_lr_beta},
                    {'params': bayes_nn.model.features.parameters()}]
            optimizer = torch.optim.Adam(parameters, lr=args.swag_lr, weight_decay=0.0)
            scheduler = ExponentialLR(optimizer, gamma=0.9)

        dt =  dtArr[min(epoch, dtArr.shape[0]-1)]
        # Number of timesteps to predict forward
        tsteps = np.zeros(len(training_loader)).astype(int) + int(90*min(epoch/75, 1) + 10)
        if(epoch >= args.swag_start):
            # Once SWAG sampling starts randomly pick a length to unroll.
            # This is done to help speed training up.
            tsteps = np.zeros(len(training_loader)).astype(int) + np.random.randint(10, int(90*min(epoch/75., 1.0) + 10), len(training_loader))
        
        # Back-prop interval
        tback = np.zeros((len(training_loader))) + np.random.randint(2,5,tsteps.shape[0])
        # Time-step to start training at
        tstart  = np.zeros(tsteps.shape[0])

        # Train network 
        loss, mse = train(args, swag_nn, burgerInt, training_loader, optimizer, \
            tsteps, tback, tstart, dt)
        print("Epoch: {}, Loss: {:0.5E}, MSE: {:0.5E}, Noise {:0.3f}" \
            .format(epoch, loss, mse, swag_nn.base.beta()))

        # If not sampling weights we can adjust the learning rate
        if (epoch < args.swag_start + 10):
            # Update the learning rate
            scheduler.step()
            for param_group in optimizer.param_groups:
                print('Epoch {}, lr: {}'.format(epoch, param_group['lr']))

        # Sample model parameters for SWAG posterior approx.
        # NOTE: 10 epoch burn in period with learning rate decay
        if(epoch >= args.swag_start + 10 and epoch % args.swag_freq == 0):
            print('Collecting model')
            swag_nn.collect()

        # Testing
        if(epoch % args.plot_freq == 0):
            n_test = 200 # Number to time-steps to test
            with torch.no_grad():
                uPred, uTarget = test(args, swag_nn.base, testing_loader, tstep=n_test)
            # Construct domain for plotting
            tTest = np.arange(0, n_test*args.dt+1e-8, args.dt)
            xTest = np.linspace(x0, x1, args.nel+1)
            for bidx in range(2):
                plotPred(args, bidx, uPred[bidx].cpu().numpy(), uTarget[bidx].cpu().numpy(), tsteps=40, \
                    target_step=4, pred_step=4, epoch=epoch)
        
            # Plot samples from swag
            if(epoch > args.swag_start):
                with torch.no_grad():
                    uPred, uTarget = testSample(args, swag_nn, testing_loader, tstep=n_test, n_samples=8)
                # Plot samples at time 0.1 and 0.25
                bidx = 0
                for t in [10, 25]:
                    plotSamples(args, bidx, uPred[bidx].detach().numpy(), uTarget[bidx].cpu().numpy(), tstep=t, epoch=epoch)

        # Save model periodically
        if(epoch % args.ckpt_freq == 0):
            swag_nn.saveModel(int(epoch), optimizer, scheduler, file_dir=args.ckpt_dir)
