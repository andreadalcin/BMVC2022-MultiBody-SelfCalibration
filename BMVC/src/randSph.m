function [X,Y,Z] = randSph(num_points)
TH = 2*pi*rand(1,num_points);
PH = asin(-1+2*rand(1,num_points));
[X,Y,Z] = sph2cart(TH,PH,1);
end