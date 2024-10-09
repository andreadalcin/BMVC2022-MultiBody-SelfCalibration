function [mu_f0, sigma_f0, x, ySix] = sturm2(Fs, width, height)

f = [];
for i = 1:size(Fs,3)
    f0 = getInitialEstimateOfFocalLength(Fs(:,:,i), width, height);
    f = [f; f0];
end

% Filter out focal lengths outside the (1e+02, 5e+05) range
f(f < 1e+02 | f > 5e+05) = [];

% Compute the KDE
bandwidth = median(f) * 0.05;
pdSix = fitdist(f,'Kernel','Width',bandwidth);

x = min(f):.1:max(f);
ySix = pdf(pdSix,x);

% Mean is the peak of the distribution
[~,I] = max(ySix);
mu_f0 = x(I);

% Standard deviation with the Q_n estimator
sigma_f0 = robstd(f, 'Q');

end