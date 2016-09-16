%expand f to [m n]
%this function can be realized by padarray, but is slower
function f=fexpand(f,mm,nn)
%size f
[m n]=size(f); 
%store f
f0=f;
%generate a larger matrix with size [mm nn]
f=zeros(mm,nn);
%copy original data
f(1:m,1:n)=f0;