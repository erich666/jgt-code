function C = drawstar(n, f)
% DRAWSTAR(N, f)
% Returns a sample two-ring of control points for face (f) adjacent
% to a boundary EV of valence N.  We draw a star shape with the 
% boundary EV at the origin.

theta = 0.5 * pi / (n-1);

p = 2*n;

if f > 0
    k = p+7;
else
    k = p+6;
end
if (n == 2)
    theta = pi * 0.25;
    k = p+5;
end

C = zeros(k, 2);

d = [1.75,1];

for i = 1:p-1
    index = rem(i,2)+1;
    C(i+1,:) = d(index) * [cos(pi-(i-1)*theta), sin(pi-(i-1)*theta)];
end

r = 2;
t = 3 + 2*f;
C(p+1,:) = C(t,:) * r;
C(p+3,:) = C(t+1,:) * r;
C(p+2,:) = (C(p+1,:) + C(p+3,:)) * 0.5;

if (n == 2)
    C(9,:) = C(2,:)*r;
    C(8,:) = (C(9,:) + C(5,:)) * 0.5;
else
    tmp = C(t+2,:) * r;
    C(p+4,:) = (tmp + C(p+3,:)) * 0.5;

    C(p+6,:) = C(t-1,:) * r;
    C(p+5,:) = (C(p+1,:) + C(p+6,:)) * 0.5;

    if f > 0
        tmp = C(t-2,:) * r;
        C(p+7,:) = (C(p+6,:) + tmp) * 0.5;
    end
end
