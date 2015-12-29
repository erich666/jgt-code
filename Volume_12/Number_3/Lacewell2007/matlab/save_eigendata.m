function save_eigendata(filename, N)
% SAVE_EIGENDATA(filename, N)
% Save eigenvalues and coefficient matrices to a file, for boundary 
% EVS with valences 2 thru N.  

cols = 3;
formatstring = '%1.16g, %1.16g, %1.16g,\n';
fid = fopen(filename, 'w');

% first save W1 for 3 cases (that's all we need): 
printW1(fid, 'inv(W1), N==2', formatstring, cols, 2, 0); 
printW1(fid, 'inv(W1), N>2, f==0', formatstring, cols, 4, 0); 
printW1(fid, 'inv(W1), N>3, f>0', formatstring, cols, 4, 1);

fprintf(fid, '//------Per-valence data starts here\n');
for n = 2:N
    p = 2*n;
    
    if (n > 3)
        [A,Abar] = subdmatrix(n,1); 
    else
        [A,Abar] = subdmatrix(n,0);
    end
    [V,D] = subdeig(A,n);
    iV = inv(V);
    d = diag(D);
    K = size(A,1);
    
    fprintf(fid, '//N = %2.0f\n', n);
    fprintf(fid, '//eigenvalues\n', n);
    fprintf(fid, formatstring, d);
    printnewline(fid, d, cols);

    U0 = iV(1:p, 1:p)';
    fprintf(fid, '//inv(U0)\n');
    fprintf(fid, formatstring, U0);
    printnewline(fid, U0, cols);
            
    for f = 0:floor(n/2)-1
        [A,Abar] = subdmatrix(n, f);
        [V,D,] = subdeig(A,n);
        iV = inv(V);
        d = diag(D);
        K = size(A,1);
        
        U1 = iV(p+1:K, 1:p)';
        fprintf(fid, '//lower left block of inv(V): face %2.0f\n', f);
        fprintf(fid, formatstring, U1);
        printnewline(fid, U1, cols);
        
        for cell = 1:3
            P = pick(n, f, cell);
            M = P*Abar*V;
            fprintf(fid, '//eigenbasis coeffs: face %2.0f cell %2.0f\n', f, cell);
            fprintf(fid, formatstring, M);
            printnewline(fid, M, cols);
        end
    end
end

fclose(fid);


function printW1(fid, comment, formatstring, cols, n, f)
[A,Abar] = subdmatrix(n,f);
[V,D] = subdeig(A,n);
iV = inv(V);
K = size(A,1);
W1 = iV(2*n+1:K, 2*n+1:K)';
fprintf(fid, '//%s\n', comment);
fprintf(fid, formatstring, W1);
printnewline(fid, W1, cols);

function printnewline(fid, M, cols)
if (mod(size(M,1)*size(M,2), cols) > 0)
    fprintf(fid, '\n');
end
