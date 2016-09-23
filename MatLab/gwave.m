%The main reason: when an image is shifted, the spectrum is modulated.

%%
% [y x]= meshgrid(-64:63, -64:63);
% w = exp(-(x.*x+y.*y)/2/20/20);
% w = w/sqrt(sum(sum(w.*w)));
% wave = w.*exp(1j*2*pi/128*20*(x+64)+1j*2*pi/128*20*(y+64));
% FW=fftshift(fft2(w));
% FWAVE=fftshift(fft2(wave));

%%
sigma = 10;
Wsize = 200;             % spatial domain window size
SF = 1;                  % spatial sampling frequency is equal to 1 (per-pixel)

dt = 1/SF;
x = -Wsize/2 : dt : Wsize/2 - dt;            % spatial grid
df = 1/Wsize;
Fmax = 1/2/dt;
f = -Fmax : df : Fmax - df;% freq grid


w = exp(-x.^2/2/sigma^2);                                  % Gaussian
Wanalytical = sigma*sqrt(2*pi)*exp(-2*pi^2*f.^2*sigma^2);  % Analytical 
Wfft = dt * fft(w);                                        % fft without shift
WIdentical = dt * fftshift(fft(fftshift(w)));              % fft with shift

hold on
plot(f,real(WIdentical),'r-', f, real(Wanalytical), 'g--', f, real(Wfft),'b-');
figure
hold on
plot(f,imag(WIdentical),'r-', f, imag(Wanalytical), 'g--', f, imag(Wfft),'b-');