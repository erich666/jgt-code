function [T,J] = subdeig(A, n)
% [T,J] = SUBDEIG(A)
% Returns the Jordan Normal Form [T,J] of subd matrix A such that 
% A = T*J*inv(T)
% For valences 3,7,11,... J is diagonal; for other valences
% J contains a single '1' on the superdiagonal

pinvtol = 10e-8;
k = size(A,1);
s = 2*n;
S = A(1:s, 1:s);
[U0, Sigma] = eig(S);
[U0,Sigma] = eigsort(U0,Sigma);

% figure out which eigenvalue of S, if any, has an incomplete set of 
% eigenvectors
m = mod(n+1, 4);
index = -1;
if m == 0
elseif m == 2
    index = (n+1)/2;
    lambda = 0.5;
else
    index = n+1;
    lambda = 0.25;
end

% build generalized eigenvector of S
if (m > 0)
    v0 = U0(:,index);
    M = (S - lambda*eye(size(S)));    
    v1 = pinv(M)*v0;
    U0(:,index+1) = v1;
    Sigma(index, index+1) = 1;
end

S11 = A(s+1:k, 1:s);
S12 = A(s+1:k, s+1:k);
[W1, Delta] = eig(S12);

% Matlab returns the boundary face eigenvals in a slightly different order
if (k - s == 6) 
   [W1,Delta] = eigswap(W1,Delta, 5, 6); 
end

C = -S11 * U0;

U1 = zeros(k-s,s);
for i = 1:s
    b = C(:,i);
    % Add an extra term to the rhs vector to account for the extra
    % '1' in the 2x2 Jordan block in Sigma
    if (i > 1 && i == index+1)
        b = b + U1(:,index);
    end

    % Note: matrix M is singular when n is odd and Sigma(i,i) == 0.125,
    % which is also an eigenvalue of S12.  In this case there are multiple 
    % solutions for U1(:,i), and the backslash operator produces a valid
    % solution and prints a warning.  pinv() gives another valid solution.
    M = (S12 - Sigma(i,i)*eye(size(S12)));
    if (mod(n,2) == 1 && i == 3*(n+1)/2)
        %U1(:,i) = M \ b;
        U1(:,i) = pinv(M, pinvtol) * b;
    else
        U1(:,i) = M \ b;   
    end
end

J = [Sigma zeros(s,k-s); zeros(k-s,s) Delta];
T = [U0 zeros(s,k-s); U1 W1];




function [V1,D1] = eigswap(V, D, index1, index2)
V1 = V;
D1 = D;
t = D(index1,index1);
D1(index1,index1) = D(index2, index2);
D1(index2, index2) = t;
v = V(:,index1);
V1(:,index1) = V(:,index2);
V1(:,index2) = v;
