function P = pick(n, f, k)
% PICK(N, F, K)
% Build a picking matrix for a boundary EV evaluation.
% N is valence, F is face ordered ccw from the left, k is cell 1,2 or 3

p = 2*n;
t = 2*f+2;

if (n == 2)
    % corner case
    P = zeros(16,16);
    ix = [0 3 6 12 1 2 5 11 8 7 4 10 15 14 13 9];
    if (k == 1)
        P(1,:) = [2 -1 zeros(1,14)];
        P(2,:) = [0 0 -1 2 zeros(1, 12)];
        P(3,:) = [0 0 0 0 0 -1 2 zeros(1, 9)];
        P(4,:) = [zeros(1,11) -1 2 0 0 0];
        for i = 1:12
            P(i+4, ix(i)+1) = 1;
        end
    elseif (k == 2)
        for i = 1:16
            P(i, ix(i)+1) = 1;
        end
    else
        P(1, :) = [2 0 0 -1 zeros(1, 12)];
        P(5, :) = [0 2 -1 zeros(1, 13)];
        P(9, :) = [0 0 0 0 0 0 0 -1 2 zeros(1, 7)];
        P(13, :)= [zeros(1, 14) -1 2];
        for i = 1:4
            for j = 1:3
                index = (i-1)*4 + j;
                P(index+1, ix(index)+1) = 1;
            end
        end
    end
else
    if f > 0
        if (k == 1)
            ix = [t+3 t+2 p+3 p+11 0 t+1 p+2 p+10 t-1 t p+1 p+9 p+5 p+4 p p+8];
        elseif (k == 2)
            ix = [0 t+1 p+2 p+10 t-1 t p+1 p+9 p+5 p+4 p p+8 p+14 p+13 p+12 p+7];
        else
            ix = [t-3 0 t+1 p+2 t-2 t-1 t p+1 p+6 p+5 p+4 p p+15 p+14 p+13 p+12];
        end

        P = zeros(16, 2*n+16);
        for i = 1:16
            P(i,ix(i)+1) = 1;
        end

    else
        if (k == 1)
            ix = [t+3 t+2 p+3 p+10 0 t+1 p+2 p+9 t-1 t p+1 p+8 p+5 p+4 p p+7];
        elseif (k == 2)
            ix = [0 t+1 p+2 p+9 t-1 t p+1 p+8 p+5 p+4 p p+7 p+13 p+12 p+11 p+6];
        else
            ix = [0 0 3 p+2 1 1 2 p+1 p+5 p+5 p+4 p p+13 p+13 p+12 p+11];
        end

        P = zeros(16, 2*n+14);
        for i = 1:16
            P(i,ix(i)+1) = 1;
        end

        if (k == 3)
            % reflect extra boundary points
            P(1,1) = 2; P(1,4) = -1;
            P(5,2) = 2; P(5,3) = -1;
            P(9,p+6) = 2; P(9,p+5) = -1;
            P(13,p+14) = 2; P(13,p+13) = -1;
        end
    end
end

    