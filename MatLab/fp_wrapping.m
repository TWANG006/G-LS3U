function p=fp_wrapping(p);
%FUNCTION
%   p=fp_wrapping(p);
%
%PURPOSE
%   Wrap a phase into (-pi,pi]
%
%INPUTS
%   p:      phase
%
%OUTPUT
%   p:      wrapped phase
%
%EXAMPLES 
%   p = fp_wrapping(p);
%
%INFO
%   Last update: 07 July 2011
%   Contact: mkmqian@ntu.edu.sg (Dr Qian Kemao)
%   Copyright reserved.

p=angle(exp(j*p));
