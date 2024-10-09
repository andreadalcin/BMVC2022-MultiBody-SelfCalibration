function [mu_f0, sigma_f0] = sturm1(Fs, width, height)

f = [];
for i = 1:size(Fs,3)
    f = [f; getInitialEstimateFromF0(Fs(:,:,i), 1e3, width, height)];
end

mu_f0 = mean(f);
sigma_f0 = sqrt(var(f));

end