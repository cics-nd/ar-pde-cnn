% Sovles the Kuramoto-Sivashinsky equation 
% with ETDRK4 scheme time-integration scheme
% Source: Unknown

% Run this file to generate simulation data
% Params:
% N: Number of discretization points
% L: Length of domain
% h: The ti
% nstp: number of time-steps


clear all;
clc;
close all;

rng(12345); % Set random seed to repeatability
N = 96;  L = 22*pi;  h = 0.1;  nstp = 6001;
a0 = zeros(N-2,1);  a0(1:4) = 0.6; % just some initial condition


ncases = 200;
for i = 1:ncases
    tic
    a0(1:4) = rand(4,1);
    [tt, at] = ksfmstp(a0, L, h, nstp, 1);
    [x, ut] = ksfm2real(at, L);
    toc
    dlmwrite(sprintf('ks_data_%d.dat', i), ut')
end

fig2 = figure('pos',[5 270 600 200],'color','w');
pcolor(tt,x,ut); shading interp; caxis([-3 3]);
title('Solution u(x,t) of Kuramoto-Sivashinsky equation, system size L = 22:');
xlabel('Time'); ylabel('x','rotat',0);
colorbar;