function [tt, aa, da] = ksfmstp(a0, d, h, nstp, np)
% Solution (with Jacobian) of Kuramoto-Sivashinsky equation 
% by ETDRK4 scheme for a given number of steps
%
% u_t = -u*u_x - u_xx - u_xxxx, periodic BCs on [-d/2, d/2]
% 
% Input:
%   a0 - initial condition, d - length parameter
%   h - stepsize,  nstp - number of steps,
%   np - output every np-th step (np = 0 - output only the final value)
% Output:
%   tt = [0:np:nstp]*h - times at which KS
%   aa - solution
%   da (optional) - Jacobian (solution of the variational problem at the final time)

%  Computation is based on v = fft(u), so linear term is diagonal.
%  Adapted from: AK Kassam and LN Trefethen, SISC 2005


  if nargin < 5, np = 0; end
  N = length(a0)+2;  Nh = N/2;  % N should be even (preferably power of 2)
  if nargout < 3,
    v = [0; a0(1:2:end-1)+1i*a0(2:2:end); 0; a0(end-1:-2:1)-1i*a0(end:-2:2)];
  else
    v = [zeros(1,N-1); [a0(1:2:end-1)+1i*a0(2:2:end) zeros(Nh-1,N-2)]; ...
         zeros(1,N-1); [a0(end-1:-2:1)-1i*a0(end:-2:2) zeros(Nh-1,N-2)]];
    v(N+2:2*N+1:end) = 1;  v(2*N+2:2*N+1:end) = 1i;
    v(2*N:2*N-1:end) = 1;  v(3*N:2*N-1:end) = -1i; end
  
  k = (2.*pi./d).*[0:Nh-1 0 -Nh+1:-1]';   % wave numbers
  L = k.^2 - k.^4;                        % Fourier multipliers
  E = exp(h*L);  E2 = exp(h*L/2);
  M = 16;                                 % no. of points for complex means
  r = exp(1i*pi*((1:M)-.5)/M);            % roots of unity
  LR = h*L(:,ones(M,1)) + r(ones(N,1),:);
  Q = h*real(mean((exp(LR/2)-1)./LR ,2));
  f1 = h*real(mean((-4-LR+exp(LR).*(4-3*LR+LR.^2))./LR.^3 ,2));
  f2 = h*real(mean((2+LR+exp(LR).*(-2+LR))./LR.^3 ,2));
  f3 = h*real(mean((-4-3*LR-LR.^2+exp(LR).*(4-LR))./LR.^3 ,2));
  aa = a0;  tt = 0;  g = 0.5i*k*N;
  if nargout < 3,
    for n = 1:nstp,
      t = n*h;                   Nv = g.*fft(real(ifft(v)).^2);
      a = E2.*v + Q.*Nv;         Na = g.*fft(real(ifft(a)).^2);
      b = E2.*v + Q.*Na;         Nb = g.*fft(real(ifft(b)).^2);
      c = E2.*a + Q.*(2*Nb-Nv);  Nc = g.*fft(real(ifft(c)).^2);
      v = E.*v + Nv.*f1 + 2*(Na+Nb).*f2 + Nc.*f3;
      if np > 0 & mod(n,np)==0 & n < nstp, 
        y1 = [real(v(2:Nh)) imag(v(2:Nh))]'; 
        aa = [aa, y1(:)];  tt = [tt, t]; end, end
    if np > 0, y1 = [real(v(2:Nh)) imag(v(2:Nh))]'; end
  else
    E =  repmat(E,1,N-1);   E2 = repmat(E2,1,N-1);  Q = repmat(Q,1,N-1);
    f1 = repmat(f1,1,N-1);  f2 = repmat(f2,1,N-1);  f3 = repmat(f3,1,N-1);
    g = [g repmat(2.*g,1,N-2)];
    for n = 1:nstp,
      t = n*h;                   rfv = real(ifft(v));   Nv = g.*fft(repmat(rfv(:,1),1,N-1).*rfv);
      a = E2.*v + Q.*Nv;         rfv = real(ifft(a));   Na = g.*fft(repmat(rfv(:,1),1,N-1).*rfv);
      b = E2.*v + Q.*Na;         rfv = real(ifft(b));   Nb = g.*fft(repmat(rfv(:,1),1,N-1).*rfv);
      c = E2.*a + Q.*(2*Nb-Nv);  rfv = real(ifft(c));   Nc = g.*fft(repmat(rfv(:,1),1,N-1).*rfv);
      v = E.*v + Nv.*f1 + 2*(Na+Nb).*f2 + Nc.*f3;
      if np > 0 & mod(n,np)==0 & n < nstp, 
        y1 = [real(v(2:Nh,1)) imag(v(2:Nh,1))]';
        aa = [aa, y1(:)]; tt = [tt, t]; end, end
    da = zeros(N-2,N-2);  da(1:2:end,:) = real(v(2:Nh,2:end));
    da(2:2:end,:) = imag(v(2:Nh,2:end));
    if np > 0, y1 = [real(v(2:Nh,1)) imag(v(2:Nh,1))]'; end
  end
  if np > 0, aa = [aa, y1(:)]; tt = [tt, h*nstp]; 
  else aa(1:2:end) = real(v(2:Nh,1));  
       aa(2:2:end) = imag(v(2:Nh,1)); tt = h*nstp; end