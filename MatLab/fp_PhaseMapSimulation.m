function [f,delta] = fp_PhaseMapSimulation(m,n)

ff = zeros(m,n,3);

d(1) = 0;
d(2) = pi/2;
d(3) = pi;

x = 1:n;
I1 = zeros(m,n);
I2 = zeros(m,n);
I3 = zeros(m,n);

for i=1:m
   I1(i,:) = 5 + 5*cos(x*0.1+d(1));
end
for i=1:m
   I2(i,:) = 5 + 5*cos(x*0.1+d(2));
end
for i=1:m
   I3(i,:) = 5 + 5*cos(x*0.1+d(3));
end

N = rand(m,n)*0.5*pi-0.25*pi;

I1 = I1 + N;
I1 = uint8((I1-min(I1(:)))/(max(I1(:))-min(I1(:)))*255);

N = rand(m,n)*0.5*pi-0.25*pi;
I2 = I2 + N;
I2 = uint8((I2-min(I2(:)))/(max(I2(:))-min(I2(:)))*255);

N = rand(m,n)*0.5*pi-0.25*pi;
I3 = I3 + N;
I3 = uint8((I3-min(I3(:)))/(max(I3(:))-min(I3(:)))*255);

for i=1:m
    for j=1:n
        ff(i,j,1) = I1(i,j);
        ff(i,j,2) = I2(i,j);
        ff(i,j,3) = I3(i,j);
    end
end

f = (ff);


% figure, imagesc(I1),colormap(gray);
% figure, imagesc(I2),colormap(gray);
% figure, imagesc(I3),colormap(gray);

delta = [d(1),d(2),d(3)];
