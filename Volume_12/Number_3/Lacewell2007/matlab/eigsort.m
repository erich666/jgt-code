function [V1,E1] = eigsort(V,E)
[e1,ix] = sort(real(diag(E)),'descend');
E1 = diag(e1);
V1 = zeros(size(V));
for i = 1:size(ix)
    V1(:,i) = real(V(:,ix(i)));
end

% occasionally Matlab finds complex-conjugate eigenvectors for
% a repeated eigenvalue.  In this case, form two real eigenvectors
% by taking the real and imaginary parts of the old ones.
% There are probably more robust detection methods, but this one works 
% for our purposes:
if (cond(V1) > 10e4)
    % look for two consecutive eigenvalues in E that have small nonzero
    % complex components.  This indicates there may be complex
    % eigenvectors.
    d = diag(E);
    foundcomplex = 0;
    for i = 1:size(d)
        if (imag(d(ix(i))) ~= 0)
            if (foundcomplex == 1)
                % found our eigenvalue
                V1(:,i) = imag(V(:,ix(i)));
                foundcomplex = 0;
            else
                foundcomplex = 1;
            end
        else
            foundcomplex = 0;
        end
    end
end
