% This matlab code is used to compare the statistical error of
% various ray tracing resulted from various sampling scheme.
% 8 gif files are assumed exist.
% control.gif  - control image, jittered sampling 20x20 samples per pixel
% hammers.gif  - Result from Hammersley p1=2, average 16 sample per pixel 
% hammers3.gif - Result from Hammersley p1=3, average 16 sample per pixel 
% halton.gif   - Result from Halton p1=2, p2=3, average 16 sample per pixel 
% halton27.gif - Result from Halton p1=2, p2=7, average 16 sample per pixel 
% jitter.gif   - Result from Jittered sampling with 4 x 4 samples per pixel
% poisson.gif  - Result from Poisson disk with d=0.2, 16 samples per pixel
% random.gif   - Result from random sampling 16 samples per pixel
% regular.gif  - Result from regular sampling 16 samples per pixel
% point.gif    - Result from regular point sampling 1 sample per pixel
clear all;
% Load the control
[c,map]=gifread('control.gif');
control=ind2gray(c,map);

% Load hammersley result p1=2
[h,map]=gifread('hammers.gif');
hammers=ind2gray(h,map);
errh = abs(control - hammers);

% Load hammersley result p1=3
[h3,map]=gifread('hammers3.gif');
hammers3=ind2gray(h3,map);
errh3 = abs(control - hammers3);

% Load halton result p1=2, p2=3
[ht,map]=gifread('halton.gif');
halton=ind2gray(ht,map);
errht = abs(control - halton);

% Load halton result p1=2, p2=7
[ht27,map]=gifread('halton27.gif');
halton27=ind2gray(ht27,map);
errht27 = abs(control - halton27);

% Load jitter result
[j,map]=gifread('jitter.gif');
jitter=ind2gray(j,map);
errj = abs(control - jitter);

%Load poisson result
[p,map]=gifread('poisson.gif');
poisson=ind2gray(p,map);
errp = abs(control - poisson);

%Load random sampling
[r,map]=gifread('random.gif');
random=ind2gray(r,map);
errr = abs(control - random);

%Load regular sampling
[rg,map]=gifread('regular.gif');
regular=ind2gray(rg,map);
errrg = abs(control - regular);

%Load point sampling
[pt,map]=gifread('point.gif');
point=ind2gray(pt,map);
errpt = abs(control - point);


% Compare the mean

mean(mean(errh))
mean(mean(errh3))
mean(mean(errht))
mean(mean(errht27))
mean(mean(errj))
mean(mean(errp))
mean(mean(errr))
mean(mean(errrg))
mean(mean(errpt))
pause;


% Compare the maximum
max(max(errh))
max(max(errh3))
max(max(errht))
max(max(errht27))
max(max(errj))
max(max(errp))
max(max(errr))
max(max(errrg))
max(max(errpt))
pause;

% Compare Standard derivation
tmp=reshape(errh,1,256*256);
std(tmp)
tmp=reshape(errh3,1,256*256);
std(tmp)
tmp=reshape(errht,1,256*256);
std(tmp)
tmp=reshape(errht27,1,256*256);
std(tmp)
tmp=reshape(errj,1,256*256);
std(tmp)
tmp=reshape(errp,1,256*256);
std(tmp)
tmp=reshape(errr,1,256*256);
std(tmp)
tmp=reshape(errrg,1,256*256);
std(tmp)
tmp=reshape(errpt,1,256*256);
std(tmp)
pause;

%Compare the RMS
tmp2 = mean(mean(errh .^2));
tmp2 ^ 0.5
tmp2 = mean(mean(errh3 .^2));
tmp2 ^ 0.5
tmp2 = mean(mean(errht .^2));
tmp2 ^ 0.5
tmp2 = mean(mean(errht27 .^2));
tmp2 ^ 0.5
tmp2 = mean(mean(errj .^2));
tmp2 ^ 0.5
tmp2 = mean(mean(errp .^2));
tmp2 ^ 0.5
tmp2 = mean(mean(errr .^2));
tmp2 ^ 0.5
tmp2 = mean(mean(errrg .^2));
tmp2 ^ 0.5
tmp2 = mean(mean(errpt .^2));
tmp2 ^ 0.5
pause;

end


