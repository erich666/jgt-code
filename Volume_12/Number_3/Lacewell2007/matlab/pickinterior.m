function P = pickinterior(n,k)
% PICKINTERIOR(N, K)
% Build a picking matrix for an interior EV evaluation.
% N is valence, k is cell 1,2 or 3

if (n < 3)
    fprintf('pickinterior: n must be at least 3\n');
    return
end

p = 2*n;
P = zeros(16,p+17);
    
if (k == 1)
    if (n == 3)
        p = [2 7 p+5 p+13 1 6 p+4 p+12 4 5 p+3 p+11 p+7 p+6 p+2 p+10];
    else
        p = [8 7 p+5 p+13 1 6 p+4 p+12 4 5 p+3 p+11 p+7 p+6 p+2 p+10];
    end
elseif (k == 2)
    p = [1 6 p+4 p+12 4 5 p+3 p+11 p+7 p+6 p+2 p+10 p+16 p+15 p+14 p+9];
else
    p = [2 1 6 p+4 3 4 5 p+3 p+8 p+7 p+6 p+2 p+17 p+16 p+15 p+14];
end

for i = 1:16
    P(i,p(i)) = 1;
end
    