function [V,D] = subdeiginterior(A, n)
% [V,D] = SUBDEIGINTERIOR(A)
% Returns the diagonalization of subd matrix A such that 
% A = V*D*inv(V)
% and D is diagonal

pinvtol = 10e-8;
k = size(A,1);
s = 2*n;
S = A(1:s+1, 1:s+1);
[U0, Sigma] = eig(S);
[U0,Sigma] = eigsort(U0,Sigma);

S11 = A(s+2:k, 1:s+1);
S12 = A(s+2:k, s+2:k);

% [W1,Delta] from Stam 98
W1 = [[1 1 2 11 1 2 11];
      [0 1 1 2  0 0 0 ];
      [0 1 0 -1 0 0 0 ];
      [0 1 -1 2 0 0 0 ];
      [0 0 0  0 1 1 2];
      [0 0 0  0 1 0 -1];
      [0 0 0  0 1 -1 2]];

Delta = diag([1/64 1/8 1/16 1/32 1/8 1/16 1/32]);

C = -S11 * U0;

U1 = zeros(k-s-1,s+1);
for i = 1:s+1
    b = C(:,i);
    M = (S12 - Sigma(i,i)*eye(size(S12)));

    if (mod(n,4) == 0 && (i == 3*n/2+1 || i == 3*n/2+2))
        U1(:,i) = pinv(M, pinvtol)*b;
    else
        U1(:,i) = M \ b;
    end
end

D = [Sigma zeros(s+1,k-s-1); zeros(k-s-1,s+1) Delta];
V = [U0 zeros(s+1,k-s-1); U1 W1];