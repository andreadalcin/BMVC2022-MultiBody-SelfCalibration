function [mu_f0, sigma_f0] = sturm1_1(Fs, width, height)

f0 = 1e3;

f = [];
for i = 1:size(Fs,3)
    f = [f; getInitialEstimateFromF0(Fs(:,:,i), f0, width, height)];
end

% % Filter focal lengths not in the peak bin
bandwidth = median(f) * 0.05;

% figure
pdSix = fitdist(f,'Kernel','Width',bandwidth);
x = min(f):.1:max(f);
ySix = pdf(pdSix,x);
% plot(x,ySix,'k-','LineWidth',2)

[~,I] = max(ySix);
mu_f0 = x(I);
sigma_f0 = robstd(f, 'Q');

% h = histogram(f, calcnbins(f,'sturges'));
% [~, whichbin] = max(h.Values);
% th_low = h.BinEdges(whichbin);
% th_high = h.BinEdges(whichbin + 1);
% f_pct = f;
% f_pct(or(f < th_low, f > th_high)) = [];
% mu_f0 = median(f_pct);
% sigma_f0 = sqrt(robustcov(f)) / 2;

end