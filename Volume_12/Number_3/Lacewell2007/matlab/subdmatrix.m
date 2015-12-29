function [A,Abar] = subdmatrix(n, f)
% SUBDMATRIX(n, f)
% build the subdivision matrix and extended matrix for a boundary EV 
% of valence n
% f is a face index starting at zero (the left boundary) and preceding ccw,
% with f < floor(n/2)

if (f >= floor(n/2))
    error('subdmatrix: face index f must be < %d for valence %d',floor(n/2), n);
end

p = 2*n;
if f > 0
    k = p+7;
else
    k = p+6;
end

% corner case
if (n == 2) 
    k = p+5;
end
    
A = zeros(k,k);

% upper left block
A(1,:) = boundaryvert(k, [0 1 p-1]);
A(2,:) = boundaryedge(k, [0 1]);
A(3,:) = face(k, [0 1 2 3]);

for i = 3:2:p-3
    A(i+1,:) = regularedge(k, [0 i i-2 i-1 i+1 i+2]);
    A(i+2,:) = face(k, [0 i i+1 i+2]);
end  
A(p,:) = boundaryedge(k, [0 p-1]);

t = 2*f + 2;

if (n == 2)
    % corner case
    A(5,:) = regularvert(9, [2  1 3 5 7  0 6 4 8]);
    A(6,:) = regularedge(9, [2 3  0 1 5 6]);
    A(7,:) = boundaryvert(9, [3 0 6]);
    A(8,:) = regularedge(9, [1 2  0 3 7 8]);
    A(9,:) = boundaryvert(9, [1 0 8]);
    B = zeros(7,9);
    B(1,:) = face(9, [4 5 2 7]);
    B(2,:) = regularedge(9, [2 5 3 6 4 7]);
    B(3,:) = face(9, [2 3 6 5]);
    B(4,:) = boundaryedge(9, [3 6]);
    B(5,:) = regularedge(9, [7 2 8 1 5 4]);
    B(6,:) = face(9, [8 1 2 7]);
    B(7,:) = boundaryedge(9, [1 8]);
else
    A(p+1,:) = regularvert(k, [t t-1 t+1 p+1 p+4 0 p p+2 p+5]);
    A(p+2,:) = regularedge(k, [t t+1 0 t-1 p+1 p+2]);
    A(p+3,:) = regularvert(k, [t+1 0 p+2 t t+2 t-1 t+3 p+1 p+3]);
    A(p+4,:) = regularedge(k, [t+1 t+2 0 t+3 p+2 p+3]);
    A(p+5,:) = regularedge(k, [t-1 t 0 t+1 p+4 p+5]);
    if f > 0
        A(p+6,:) = regularvert(k, [t-1 0 t-2 t p+5 t-3 t+1 p+4 p+6]);
        A(p+7,:) = regularedge(k, [t-2 t-1 0 t-3 p+5 p+6]);
    else
        A(p+6,:) = boundaryvert(k, [1 0 p+5]);
    end
    B = zeros(k-p+2, k);
    B(1,:) = face(k, [p p+1 t p+4]);
    B(2,:) = regularedge(k, [t p+1 t+1 p+2 p p+4]);
    B(3,:) = face(k, [t t+1 p+1 p+2]);
    B(4,:) = regularedge(k, [t+1 p+2 t p+1 t+2 p+3]);
    B(5,:) = face(k, [t+1 t+2 p+2 p+3]);
    B(6,:) = regularedge(k, [t p+4 t-1 p+5 p p+1]);
    B(7,:) = face(k, [t-1 t p+4 p+5]);
    if f > 0
        B(8,:) = regularedge(k, [t-1 p+5 t-2 t p+4 p+6]);
        B(9,:) = face(k, [t-2 t-1 p+5 p+6]);
    else
        B(8,:) = boundaryedge(k, [t-1 p+5]);
    end
end

Abar = [A; B];


% Subfunctions for building one row of the subdivision matrix
% using Catmull-Clark rules
% Note: all indices are zero-based

function r = face(n, indices)
% 4 indices
% 
%  0-----0
%  |     |
%  |  x  |
%  |     |
%  0-----0
%
r = zeros(1,n);
for i = 1:4
    r(indices(i)+1) = 1/4;
end


function r = regularvert(n, indices)
% 9 indices
%
%  2---1---2
%  |   |   |
%  1---0---1
%  |   |   |
%  2---1---2
%
r = zeros(1, n);
r(indices(1)+1) = 9/16;
for i = 2:5
    r(indices(i)+1) = 3/32;
end
for i = 6:9
    r(indices(i)+1) = 1/64;
end


function r = boundaryvert(n, indices)
% 3 indices:  
%  1---0---1
r = zeros(1,n);
r(indices(1)+1) = 6/8;
r(indices(2)+1) = 1/8;
r(indices(3)+1) = 1/8;


function r = regularedge(n, indices)
% 6 indices ('x' marks new point)
%
%  1-----1
%  |     |
%  0--x--0
%  |     |
%  1-----1
%
r = zeros(1,n);
r(indices(1)+1) = 3/8;
r(indices(2)+1) = 3/8;
for i = 3:6
    r(indices(i)+1) = 1/16;
end


function r = boundaryedge(n, indices)
% 2 indices: 
%  0---x---0
r = zeros(1, n);
r(indices(1)+1) = 1/2;
r(indices(2)+1) = 1/2;


