% function f0 = getInitialEstimateOfFocalLength(F, width, height)
% f0 = 2 * (width + height);
% f0 = getInitialEstimateFromF0(F, f0, width, height);
% focal_lengths = nonzeros(real(focal_lengths));
% focal_lengths = focal_lengths(~isnan(focal_lengths));
% end
function fs = getInitialEstimateOfFocalLength(F, width, height)

load('BMVC/src/sturm.mat', 'opening_angles')

fs = [];

for i = 1:size(opening_angles,2)
    f0 = max(width, height) / 2 / tan(deg2rad(opening_angles(i) / 2));
    fs = [fs; getInitialEstimateFromF0(F, f0, width, height)];
end

% f0 = 10^3;
% f0 = 2 * (width + height);
% fs = getInitialEstimateFromF0(F, f0, width, height);

fs = nonzeros(real(fs));
fs = fs(~isnan(fs));
end