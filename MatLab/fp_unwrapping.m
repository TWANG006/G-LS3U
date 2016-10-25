function p=fp_unwrapping(p,start_x,start_y)
%FUNCTION
%   p=fp_unwrapping(p,start_x,start_y)
%
%PURPOSE
%   Line scanning phase unwrapping
%
%INPUTS
%   p:          wrapped phasethe phase distribution
%   start_x:    starting point for phase unwrapping, x
%   start_y:    starting point for phase unwrapping, y
%
%OUTPUT
%   p:          unwrapped phase
%
%EXAMPLES 
%   p = fp_unwrapping(p); for (1D & 2D)
%   p = fp_unwrapping(p,100);  (for 1D)
%   p = fp_unwrapping(p,100,100); (for 2D)
%
%INFO
%   Last update: 30 June 2011
%   Contact: mkmqian@ntu.edu.sg (Dr Qian Kemao)
%   Copyright reserved.


if min(size(p))==1
    if nargin==2
        p=fp_unwrapping1(p,start_x);
    else
        p=fp_unwrapping1(p);
    end
else
    if nargin==3
        p=fp_unwrapping2(p,start_x,start_y);
    else
        p=fp_unwrapping2(p);
    end
end
    

function p=fp_unwrapping1(p,start_x)
%1D Phase Unwrapping.

if nargin==1
    start_x=round(length(p)/2);
end    

for i=start_x+1:length(p)
    p(i)=p(i)+round((p(i-1)-p(i))/2/pi)*2*pi;
end

for i=start_x-1:-1:1
    p(i)=p(i)+round((p(i+1)-p(i))/2/pi)*2*pi;
end

function p=fp_unwrapping2(p,start_x,start_y)
%2D Phase Unwrapping, using line by line scanning

[m n]=size(p);
if nargin==1
    start_x=round(m/2);
    start_y=round(n/2);
end    

%seed column is unwrapped
p(:,start_y)=fp_unwrapping1(p(:,start_y),start_x);
%all rows are unwrapped one by one
for i=1:m
    p(i,:)=fp_unwrapping1(p(i,:),start_y);
end
