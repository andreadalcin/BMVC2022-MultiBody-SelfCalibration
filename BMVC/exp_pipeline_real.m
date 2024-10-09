clc, clear, close all
format longG;

addpath('ECCV/src')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Choose the dataset

% dataset = "fountain";
% dataset = "herzjesu";
% dataset = "castle";
dataset = "S1-Amiibo";
% dataset = "S2-Amiibo";
% dataset = "S3-Amiibo";
% dataset = "M1-Amiibo";
% dataset = "M2-Amiibo";
% dataset = "M3-Amiibo";
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

switch dataset
    case "fountain"
        load('Dataset/optimization/fountain.mat')
        width = 3072;
        height = 2048;
        fx_gt = 2759.48;
        fy_gt = 2764.16;
        u_gt = 1520.69;
        v_gt = 1006.81;
    case "herzjesu"
        load('Dataset/optimization/herzjesu.mat')
        width = 3072;
        height = 2048;
        fx_gt = 2759.48;
        fy_gt = 2764.16;
        u_gt = 1520.69;
        v_gt = 1006.81;
    case "castle"
        load('Dataset/optimization/castle.mat')
        width = 3072;
        height = 2048;
        fx_gt = 2759.48;
        fy_gt = 2764.16;
        u_gt = 1520.69;
        v_gt = 1006.81;
    case "S1-Amiibo"
        load('Dataset/optimization/01_amiibo_static.mat')
        load('Dataset/raw/params.mat')
        width = cameraParams.ImageSize(2);
        height = cameraParams.ImageSize(1);
        fx_gt = cameraParams.Intrinsics.FocalLength(1);
        fy_gt = cameraParams.Intrinsics.FocalLength(2);
        u_gt = cameraParams.Intrinsics.PrincipalPoint(1);
        v_gt = cameraParams.Intrinsics.PrincipalPoint(2);
    case "S2-Amiibo"
        load('Dataset/optimization/05_amiibo_static.mat')
        load('Dataset/raw/params.mat')
        width = cameraParams.ImageSize(2);
        height = cameraParams.ImageSize(1);
        fx_gt = cameraParams.Intrinsics.FocalLength(1);
        fy_gt = cameraParams.Intrinsics.FocalLength(2);
        u_gt = cameraParams.Intrinsics.PrincipalPoint(1);
        v_gt = cameraParams.Intrinsics.PrincipalPoint(2);
    case "S3-Amiibo"
        load('Dataset/optimization/06_amiibo_static.mat')
        load('Dataset/raw/params.mat')
        width = cameraParams.ImageSize(2);
        height = cameraParams.ImageSize(1);
        fx_gt = cameraParams.Intrinsics.FocalLength(1);
        fy_gt = cameraParams.Intrinsics.FocalLength(2);
        u_gt = cameraParams.Intrinsics.PrincipalPoint(1);
        v_gt = cameraParams.Intrinsics.PrincipalPoint(2);
    case "M1-Amiibo"
        load('Dataset/optimization/02_amiibo_motion.mat')
        load('Dataset/raw/params.mat')
        width = cameraParams.ImageSize(2);
        height = cameraParams.ImageSize(1);
        fx_gt = cameraParams.Intrinsics.FocalLength(1);
        fy_gt = cameraParams.Intrinsics.FocalLength(2);
        u_gt = cameraParams.Intrinsics.PrincipalPoint(1);
        v_gt = cameraParams.Intrinsics.PrincipalPoint(2);
    case "M2-Amiibo"
        load('Dataset/optimization/10_amiibo_motion.mat')
        load('Dataset/raw/params.mat')
        width = cameraParams.ImageSize(2);
        height = cameraParams.ImageSize(1);
        fx_gt = cameraParams.Intrinsics.FocalLength(1);
        fy_gt = cameraParams.Intrinsics.FocalLength(2);
        u_gt = cameraParams.Intrinsics.PrincipalPoint(1);
        v_gt = cameraParams.Intrinsics.PrincipalPoint(2);
    case "M3-Amiibo"
        load('Dataset/optimization/12_amiibo_motion.mat')
        load('Dataset/raw/params.mat')
        width = cameraParams.ImageSize(2);
        height = cameraParams.ImageSize(1);
        fx_gt = cameraParams.Intrinsics.FocalLength(1);
        fy_gt = cameraParams.Intrinsics.FocalLength(2);
        u_gt = cameraParams.Intrinsics.PrincipalPoint(1);
        v_gt = cameraParams.Intrinsics.PrincipalPoint(2);
end

F_x = [];
F_y = [];
U = [];
V = [];

num_runs = 1;

for i = 1:num_runs
    [fx, fy, u, v] = self_calibrate(Fs, width, height);
    F_x = [F_x; fx];
    F_y = [F_y; fy];
    U = [U; u];
    V = [V; v];
end

fx = median(F_x);
fy = median(F_y);
u = median(U);
v = median(V);

err_f = 0.5 * 100 * (abs((fx - fx_gt)/fx_gt) + abs((fy - fy_gt)/fy_gt));
err_pp = 0.5 * 100 * (abs((u - u_gt)/u_gt) + abs((v - v_gt)/v_gt));

disp("Relative Error (%) - Focal length")
disp(err_f)
disp("Relative Error (%) - Principal point")
disp(err_pp)


function [fx, fy, u, v] = self_calibrate(Fs, width, height)
tic

weights = ones(size(Fs,3),1);
weights = weights ./ sum(weights);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Estimate the initial focal length
[mu_f0, sigma_f0] = sturm2(Fs, width, height);
fprintf("Peak of KDE: %f\n", mu_f0)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
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
[~, index] = sort(R);
F_min = Fs(:,:,index(1:3));

% Refinement with F_min
[fx, fy, u, v] = optimizePrincipalPoints(F_min, f, width, height);

toc
end


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