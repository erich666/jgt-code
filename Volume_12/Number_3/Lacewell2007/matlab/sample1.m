% sample script to show how to build a subdivision matrix, 
% find its Jordan Normal Form [T,J], and subdivide some sample 
% control points

[A,Abar] = subdmatrix(4,1);
[T,J] = subdeig(A,4);

% create and plot some control points
C = drawstar(4,1);
figure('Name', 'Original Control Points');
scatter(C(:,1), C(:,2))
axis equal;

% plot subdivided control points
figure('Name', 'Subdivided Control Points');
Cp = T*J^3*inv(T)*C;
scatter(Cp(:,1), Cp(:,2))
axis equal;
