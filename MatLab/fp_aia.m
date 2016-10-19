function [phi delta iter err] = fp_aia(f, delta, max_iter, max_err)
%FUNCTION
%   [phi delta iter err]=aia(f, delta, max_iter, max_err)
%
%PURPOSE
%   Wang and Han's AIA for phase-shifting with arbitrary phase-shifts.
%
%INPUT
%   f:          3-D matrix of size m*n*l (image size m*n, frame number l)
%   delta:      initial guess of phase-shift vector (k*1), default value: random 
%   max_iter:   maximum iteration number, default value: 20
%   max_err:    maximum tolorable error, default value: 10^-4
%
%OUTPUT
%   phi:        retrieved phase
%   delta:      retrieved phase-shifts
%   iter:       iteration number 
%   err:        error which is the difference of delta between two iterations
%
%EXAMPLES   
%   [phi delta iter err] = aia(f)
%
%REFERENCE
%[1]Z Wang and B Han, "Advanced iterative algorithm for phase
%extraction of randomly phase-shifted interferograms", Optics Letters, 29,
%1671-1673 (2004)
%
%INFO
%   Update history: 16 May 2012
%   Contact: mkmqian@ntu.edu.sg (Dr Qian Kemao)
%   Copyright reserved.

%image size
[m n l] = size(f);
%default parameter setting
if nargin == 1
    delta = randn(l,1); max_iter = 20; max_err = 10^-4;
elseif nargin == 2
    max_iter = 20; max_err = 10^-4;
elseif nargin == 3
    max_err = 10^-4;
end
%inital value of iter
iter = 0;
%inital value of err
err = max_err*2;

delta

while err>max_err & iter<max_iter 

    %save delta value
    delta_old = delta;
    %step 1
    phi = phi_est(f,delta);
    %step 2
    delta = delta_est(f,phi);
    %update
    iter=iter+1;
    err=max(abs(delta-delta_old));
    
end

%One more round to caluclate phi once delta is good enough
phi=phi_est(f,delta);

%If we assume that delta(1)=0;
phi=phi+delta(1);
phi=angle(exp(sqrt(-1)*phi));
delta=delta-delta(1);
delta=angle(exp(sqrt(-1)*delta));

function phi=phi_est(f,delta)

[m n l]=size(f);

A1(1,1)=l;
A1(1,2)=sum(cos(delta));
A1(1,3)=sum(sin(delta));
A1(2,1)=A1(1,2);
A1(2,2)=sum(cos(delta).*cos(delta));
A1(2,3)=sum(cos(delta).*sin(delta));
A1(3,1)=A1(1,3);
A1(3,2)=A1(2,3);
A1(3,3)=sum(sin(delta).*sin(delta));
A1_inv=pinv(A1);

for i=1:m
    for j=1:n
        t=f(i,j,:);
        B1(1,1)=sum(t(:));
        B1(2,1)=sum(t(:).*cos(delta));
        B1(3,1)=sum(t(:).*sin(delta));
        X1=A1_inv*B1;
        phi(i,j)=atan2(-X1(3),X1(2));
    end
end

function delta=delta_est(f,phi)

[m n l]=size(f);

A2(1,1)=m*n;
A2(1,2)=sum(sum(cos(phi)));
A2(1,3)=sum(sum(sin(phi)));
A2(2,1)=A2(1,2);
A2(2,2)=sum(sum(cos(phi).*cos(phi)));
A2(2,3)=sum(sum(cos(phi).*sin(phi)));
A2(3,1)=A2(1,3);
A2(3,2)=A2(2,3);
A2(3,3)=sum(sum(sin(phi).*sin(phi)));
A2_inv=pinv(A2);

for k=1:l
    t=f(:,:,k);
    B2(1,1)=sum(sum(t));
    B2(2,1)=sum(sum(t.*cos(phi)));
    B2(3,1)=sum(sum(t.*sin(phi)));
    X2=A2_inv*B2;
    delta(k,1)=atan2(-X2(3),X2(2));
end
