function [x, u, ux, uxx] = ksfm2real(a, L, n)
%  [X, U, UX, UXX] = KSFM2REAL(A, L, N)
%  Convert FM representation A(t) to real space U(X,t).  
%  Also calculate U'(X,t) and U''(X,t) (optional).
%  If N is specified and SIZE(A,1) < N+2, then pad A with zeros before
%  transforming to real space.

  if nargin < 3, n = size(a,1)+2; end
  if n < size(a,1)+2, n = size(a,1)+2; end
  nt = size(a,2); x = L.*(-n/2:n/2)'./n; v = a(1:2:end,:) + 1i*a(2:2:end,:);
  vv = [zeros(1,nt); v; zeros(n-size(a,1)-1,nt); flipud(conj(v))];
  u = real(fft(vv)); u = [u; u(1,:)];
  
  if nargout > 2, 
    ik = -(2i*pi/L)*(1:size(v,1))';  vx = repmat(ik,1,nt).*v;
    vv = [zeros(1,nt); vx; zeros(n-size(a,1)-1,nt); flipud(conj(vx))];
    ux = real(fft(vv)); ux = [ux; ux(1,:)]; end
  
  if nargout > 3,
    vxx = repmat(ik,1,nt).*vx;
    vv = [zeros(1,nt); vxx; zeros(n-size(a,1)-1,nt); flipud(conj(vxx))];
    uxx = real(fft(vv)); uxx = [uxx; uxx(1,:)]; end