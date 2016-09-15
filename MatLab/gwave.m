%The main reason: when an image is shifted, the spectrum is modulated.

%%
[y x]= meshgrid(-64:63, -64:63);
w = exp(-(x.*x+y.*y)/2/20/20);
w = w/sqrt(sum(sum(w.*w)));
wave = w.*exp(1j*2*pi/128*20*(x+64)+1j*2*pi/128*20*(y+64));
FW=fftshift(fft2(w));
FWAVE=fftshift(fft2(wave));

