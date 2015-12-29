function save_eigendata_interior(filename, N)
% SAVE_EIGENDATA_INTERIOR(filename, N)
% Save eigenvalues and coefficient matrices to a file, for interior
% EVS with valences 3 thru N. 

cols = 3;
formatstring = '%1.16g, %1.16g, %1.16g,\n';
fid = fopen(filename, 'w');

for n = 3:N
    p = 2*n;
    [A,Abar] = subdmatrixinterior(n);
    [V,D] = subdeiginterior(A,n);
    d = diag(D);
    
    fprintf(fid, '//N = %2.0f\n', n);
    fprintf(fid, '//eigenvalues\n', n);
    fprintf(fid, formatstring, d);
    printnewline(fid, d, cols);

    iV = inv(V)';
    fprintf(fid, '//inv(V)\n');
    fprintf(fid, formatstring, iV);
    printnewline(fid, iV, cols);
     
    for cell = 1:3
        P = pickinterior(n, cell);
        M = P*Abar*V;
        fprintf(fid, '//eigenbasis coeffs: cell %2.0f\n', cell);
        fprintf(fid, formatstring, M);
        printnewline(fid, M, cols);
    end
   
end

fclose(fid);


function printnewline(fid, M, cols)
if (mod(size(M,1)*size(M,2), cols) > 0)
    fprintf(fid, '\n');
end
