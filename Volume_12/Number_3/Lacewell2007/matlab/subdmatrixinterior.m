function [A,Abar] = subdmatrixinterior(n)
% SUBDMATRIXINTERIOR(n)
% build the subdivision matrix and extended matrix for a boundary EV
% of valence n.  This is straight from Stam 98, Appendix A.

p = 2*n;
aN = 1 - 7/(4*n);
bN = 3/(2*n*n);
cN = 1/(4*n*n);
d = 3/8;
e = 1/16;
f = 1/4;

S = zeros(p+1,p+1);
S(1,1) = aN;
for i = 2:2:p
    S(1,i) = bN;
    S(1,i+1) = cN;
end

S(2,:) = [d d e e zeros(1,p-5) e e];
S(3,:) = [f f f f zeros(1, p-3)];

for i = 4:2:p-2
    S(i,1) = d;
    S(i+1,1) = f;
    index = (i-2);
    S(i,index:index+4) = [e e d e e];
    S(i+1,i:i+2) = [f f f];
end

S(p,:) = [d e zeros(1,p-5) e e d e];
S(p+1,:) = [f f zeros(1,p-3) f f];

a = 9/16; b = 3/32; c = 1/64;

S12 = [[c b c 0 b c 0];
       [0 e e 0 0 0 0];
       [0 c b c 0 0 0];
       [0 0 e e 0 0 0];
       [0 0 0 0 e e 0];
       [0 0 0 0 c b c];
       [0 0 0 0 0 e e]];
   
Z = zeros(1,p-7);
S11 = [[c 0 0 b a b 0 0 Z];
       [e 0 0 e d d 0 0 Z];
       [b 0 0 c b a b c Z];
       [e 0 0 0 0 d d e Z];
       [e 0 0 d d e 0 0 Z];
       [b c b a b c 0 0 Z];
       [e e d d 0 0 0 0 Z]];

if (n == 3)
    S11(:,2) = [0 0 c e 0 c e]';
    S11 = S11(:,1:7);
end

A = [S zeros(p+1,7); S11 S12];

Z = zeros(1, p-6);
S21 = [[0 0 0 0 f 0 0 Z];
       [0 0 0 0 d e 0 Z];
       [0 0 0 0 f f 0 Z];
       [0 0 0 0 e d e Z];
       [0 0 0 0 0 f f Z];
       [0 0 0 e d 0 0 Z];
       [0 0 0 f f 0 0 Z];
       [0 0 e d e 0 0 Z];
       [0 0 f f 0 0 0 Z]];
   
S22 = [[f f 0 0 f 0 0];
       [e d e 0 e 0 0];
       [0 f f 0 0 0 0];
       [0 e d e 0 0 0];
       [0 0 f f 0 0 0];
       [e e 0 0 d e 0];
       [0 0 0 0 f f 0];
       [0 0 0 0 e d e];
       [0 0 0 0 0 f f]];
   
Abar = [A; S21 S22];
   
   
   