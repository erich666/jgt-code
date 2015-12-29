function C = drawstarinterior(n)
% DRAWSTARINTERIOR(N)
% Returns a sample two-ring of control points for an interior EV.  

theta = pi / n;

C = zeros(2*n+8,2);

d = [1.9,1];

for i = 1:2*n
    index = rem(i,2)+1;
    C(i+1,:) = d(index) * [cos(pi-(i-1)*theta), sin(pi-(i-1)*theta)];
end

p = 2*n;
C(p+3,:) = 2*C(5,:) - C(4,:);
C(p+4,:) = 2*C(6,:) - C(1,:);
C(p+5,:) = 2*C(7,:) - C(8,:);
C(p+2,:) = 2*C(p+3,:) - C(p+4,:);

C(p+6,:) = 2*C(5,:) - C(6,:);
C(p+7,:) = 2*C(4,:) - C(1,:);
C(p+8,:) = 2*C(3,:) - C(2,:);