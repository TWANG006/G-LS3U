function dphiSum = fp_ls3u(phi0, fname, startNo, endNo, ext, rr, f0)
%FUNCTION
%   dphiSum = fp_ls3u(phi0, fname, startNo, endNo, ext, rr, f0)
%
%PURPOSE
%   Dynamic phase retrieval by LS3U and WFF2.
%
%INPUT
%   phi0:       3-D matrix of size m*n*l (image size m*n, frame number l)
%   fname:      the name of file sequence, the real name should be fnameXXXX 
%   startNo:    start number of XXXX
%   endNo:      end number of XXXX
%   ext:        extension of files, default: bmp
%   rr:         reference rate, defulat: 1
%   f0:         the reference frame if you want to generate speckle
%               correlation fringe pattern
%
%OUTPUT
%   dphiSum:    demoudlated phase
%
%EXAMPLES   
%   phi = fp_ls3u(phi0, 'test', 1, 100, 'bmp', 1);
%
%REFERENCE
%[1]L. Kai and Q. Kemao, “Dynamic phase retrieval in temporal speckle 
%   pattern interferometry using least squares method and windowed Fourier
%   filtering,? Opt. Express 19, 18058-18066 (2011).
%[2]L. Kai and Q. Kemao, “Dynamic 3D profiling with fringe projection using
%   least squares method and windowed Fourier filtering,?Optics and Lasers
%   in Engineering 51, 1-7 (2013).
%
%INFO
%   Update history: 16/05/2012, 25/05/2012
%   Contact: mkmqian@ntu.edu.sg (Dr Qian Kemao)
%   Copyright reserved.


%set default values
if nargin == 4
    ext = 'bmp'; rr = 1;
elseif nargin == 5
    rr = 1;
%this part is to generate speckle correlation fringe pattern
elseif nargin == 7
    for k = startNo: endNo
        %read image
        name = [fname num2str(k) '.' ext];
        f = double(imread(name));
        %generate speckle correlation fringe pattern
        sc = abs(f-f0);
        name = ['sc' num2str(k) '.' ext];
        imwrite(uint8(sc*5), name);
    end
end

%Initialization
[m n] = size(phi0);
jj = sqrt(-1);
%reference phase
phiRef = phi0;
%reference phase change
dphiRef = phi0*0; 
X=0;
for k = startNo: endNo
    k
    tic;
    %read image
    name = [fname num2str(k) '.' ext];
    f = double(imread(name));
    %main part of LS3U, window size is 3*3
    for i = 1:m
        for j= 1:n
            t = phiRef(max(i-1,1):min(i+1,m),max(j-1,1):min(j+1,n));
            c = cos(t(:)); s = sin(t(:));
            ft = f(max(i-1,1):min(i+1,m),max(j-1,1):min(j+1,n));
            A(1,1) = 9;
            A(1,2) = sum(c);
            A(1,3) = sum(s);
            A(2,1) = A(1,2);
            A(2,2) = sum(c.^2);
            A(2,3) = sum(c.*s);
            A(3,1) = A(1,3);
            A(3,2) = A(2,3);
            A(3,3) = sum(s.^2);
            B(1,1) = sum(ft(:));
            B(2,1) = sum(ft(:).*c);
            B(3,1) = sum(ft(:).*s);
            X = A\B;
            dphi(i,j) = atan2(-X(3),X(2));
        end
    end
    %denoising by WFF2
    dphiWft = exp(sqrt(-1)*dphi);
    
    fb=0.2;
    t = fp_wft2f('wff',dphiWft, 20,-fb,0.1,fb,20,-fb,0.1,fb,15);
%    t = fp_wft2f('wff',exp(sqrt(-1)*dphi),10,-fb,0.1,fb,10,-fb,0.1,fb,5);

    dphi = angle(t.filtered);
    %update the phase
    phi = angle(exp(jj*(phiRef + dphi)));
    %update the phase change wrt initial status
    k0 = k-startNo+1;
    dphiSum(:,:,k0) = dphiRef+fp_unwrapping(dphi,20,20);
   
    %update reference
    if mod(k0, rr) == 0
        phiRef = phi;
        dphiRef = dphiSum(:,:,k0);
    end
    toc
end

dpmax = max(dphiSum(:));
dpmin = min(dphiSum(:));
for k = startNo: endNo
    k0 = k-startNo+1;
    
    %save phase change unwrapped
    name = ['dp' num2str(k) '.' ext];
    t0 = uint8((dphiSum(:,:,k0)-dpmin)/(dpmax-dpmin)*255);
    imwrite(t0, name);
    %for video
    t1(:,:,1) = t0;t1(:,:,2)=t0;t1(:,:,3)=t0;
    dp(k0) = im2frame(t1);

    %save phase change wrapped
    name = ['dpw' num2str(k) '.' ext];                         
    t0 = uint8((fp_wrapping(dphiSum(:,:,k0))+pi)/2/pi*255);
    imwrite(t0, name);
    %for video
    t1(:,:,1) = t0;t1(:,:,2)=t0;t1(:,:,3)=t0;
    dpw(k0) = im2frame(t1);
end  
movie2avi(dp, 'dp.avi', 'compression', 'None');
movie2avi(dpw, 'dpw.avi', 'compression', 'None');

