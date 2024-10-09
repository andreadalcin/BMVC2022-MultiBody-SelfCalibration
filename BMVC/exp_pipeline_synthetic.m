clc, clear, close all
format longG;

load('Dataset/synthetic/camera.mat')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Choose testing config
% Filename format => test-{num_cameras}-{num_motions}

% load('Dataset/synthetic/test-3-1.mat')
load('Dataset/synthetic/test-3-2.mat')
% load('Dataset/synthetic/test-3-3.mat')
% load('Dataset/synthetic/test-5-1.mat')
% load('Dataset/synthetic/test-5-2.mat')
% load('Dataset/synthetic/test-5-3.mat')
% load('Dataset/synthetic/test-7-1.mat')
% load('Dataset/synthetic/test-7-2.mat')
% load('Dataset/synthetic/test-7-3.mat')
% load('Dataset/synthetic/test-10-1.mat')
% load('Dataset/synthetic/test-10-2.mat')
% load('Dataset/synthetic/test-10-3.mat')
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Estimate the initial focal length
[mu_f0, sigma_f0] = sturm2(Fs, width, height);
disp("Robust initialization")
fprintf("mu: %f, sigma: %f\n", mu_f0, sigma_f0)

% Optimization
F = [];
N = size(Fs,3);
max_iters = min(log(1 - 0.95) / log(1 - 1/nchoosek(N,3)), 10^4);

parfor i = 1:round(max_iters)
    [f, ~] = optimizeFocalLength(Fs, mu_f0, sigma_f0, width, height);
    F(i) = f;
end

f = kernel_voting(F', 0.01);

% Find minimum residuals
R = zeros(N,1);
for i = 1:N
    X = [f f width/2 height/2];
    R(i) = costFunctionMendoncaCipolla(Fs(:,:,i), X, '2');
end
[xs, index] = sort(R);
F_min = Fs(:,:,index(1:3));

% Refinement with F_min
[fx, fy, u, v] = optimizePrincipalPoints(F_min, f, width, height);

fprintf("Focal lengths\t= (%f, %f)\n", fx, fy);
fprintf("Principal point\t= (%f, %f)\n", u, v);


function f = kernel_voting(f, bandwidth_factor)
f(f < 100 | f > 1e5) = [];
bandwidth = median(f) * bandwidth_factor;
pdSix = fitdist(f,'Kernel','Width',bandwidth);
x = min(f):.1:max(f);
ySix = pdf(pdSix,x);
[~,I] = max(ySix);
f = x(I);
% figure
% plot(x,ySix)
end


function [f,views] = optimizeFocalLength(Fs, mu_f0, sigma_f0, width, height)

% Randomize focal length and principal point
f0 = 0;
u0 = 0;
v0 = 0;

while (f0 <= 10^2 || f0 >= 10^5)
    f0 = normrnd(mu_f0, sigma_f0);
end
while (u0 <= 0)
    u0 = normrnd(width / 2, width / 6);
end
while (v0 <= 0)
    v0 = normrnd(height / 2, height / 6);
end

K0 = [f0, 0, u0; 0, f0, v0; 0, 0, 1];

% Optimization
Options = optimoptions('lsqnonlin','Display','off', ...
    'Algorithm','levenberg-marquardt', ...
    'StepTolerance',1e-20,...
    'FunctionTolerance',1e-20,...
    'MaxIterations',1e2,...
    'MaxFunctionEvaluations',1e6,...
    'TolFun', 1e-20,...
    'TolX',1e-20);

% Randomize focal length
X0 = [f0];

views = datasample(1:size(Fs,3),3,'Replace',false);
loss = @(X) costFunctionMendoncaCipollaFocalOnly(Fs(:,:,views), X, u0, v0, '2');

K_SK = lsqnonlin(loss, X0, [], [], Options);
K_SK = [K_SK(1) 0 width/2; 0 K_SK(1) height/2; 0 0 1];

f = K_SK(1,1);
end


function [fx,fy,u,v] = optimizePrincipalPoints(F_min, f, width, height)
u0 = width / 2;
v0 = height / 2;
K0 = [f, 0, u0; 0, f, v0; 0, 0, 1];

% Optimization
Options = optimoptions('lsqnonlin','Display','off', ...
    'Algorithm','levenberg-marquardt', ...
    'StepTolerance',1e-20,...
    'FunctionTolerance',1e-20,...
    'MaxIterations',10,...
    'MaxFunctionEvaluations',1e6,...
    'TolFun', 1e-20,...
    'TolX',1e-20);

% Randomize focal length
X0 = [K0(1,1) K0(2,2) K0(1,3), K0(2,3)];

loss = @(X) costFunctionMendoncaCipolla(F_min, X, '2');

K_SK = lsqnonlin(loss, X0, [], [], Options);
K_SK = [K_SK(1) 0 K_SK(3); 0 K_SK(2) K_SK(4); 0 0 1];

fx = K_SK(1,1);
fy = K_SK(2,2);
u = K_SK(1,3);
v = K_SK(2,3);
end