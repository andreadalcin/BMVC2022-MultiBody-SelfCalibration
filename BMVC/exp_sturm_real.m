clc, clear, close all
format longG;

rng default;

addpath('BMVC/src/')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Choose the dataset

% dataset = "fountain";
% dataset = "herzjesu";
% dataset = "castle";
% dataset = "S1-Amiibo";
dataset = "S2-Amiibo";
% dataset = "S3-Amiibo";
% dataset = "M1-Amiibo";
% dataset = "M2-Amiibo";
% dataset = "M3-Amiibo";
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

xaxis_limit = true;

switch dataset
    case "fountain"
        load('Dataset/sturm/fountain.mat')
        width = 3072;
        height = 2048;
        f_gt = 2759.48;
        Fs = pre_process(Fs);
    case "castle"
        load('Dataset/sturm/castle.mat')
        width = 3072;
        height = 2048;
        f_gt = 2759.48;
        xaxis_limit = false;
        Fs = pre_process(Fs);
    case "herzjesu"
        load('Dataset/sturm/herzjesu.mat')
        width = 3072;
        height = 2048;
        f_gt = 2759.48;
        Fs = pre_process(Fs);
    case "S1-Amiibo"
        load('Dataset/sturm/01_amiibo_static.mat')
        load('Dataset/raw/params.mat')
        width = cameraParams.Intrinsics.ImageSize(2);
        height = cameraParams.Intrinsics.ImageSize(1);
        f_gt = mean(cameraParams.Intrinsics.FocalLength);
        Fs = pre_process(Fs);
    case "M1-Amiibo"
        load('Dataset/sturm/02_amiibo_motion.mat')
        load('Dataset/raw/params.mat')
        width = cameraParams.Intrinsics.ImageSize(2);
        height = cameraParams.Intrinsics.ImageSize(1);
        f_gt = mean(cameraParams.Intrinsics.FocalLength);
        Fs = pre_process(Fs);
    case "S2-Amiibo"
        load('Dataset/sturm/05_amiibo_static.mat')
        load('Dataset/raw/params.mat')
        width = cameraParams.Intrinsics.ImageSize(2);
        height = cameraParams.Intrinsics.ImageSize(1);
        f_gt = mean(cameraParams.Intrinsics.FocalLength);
        Fs = pre_process(Fs);
    case "S3-Amiibo"
        load('Dataset/sturm/06_amiibo_static.mat')
        load('Dataset/raw/params.mat')
        width = cameraParams.Intrinsics.ImageSize(2);
        height = cameraParams.Intrinsics.ImageSize(1);
        f_gt = mean(cameraParams.Intrinsics.FocalLength);
        Fs = pre_process(Fs);
    case "M2-Amiibo"
        load('Dataset/sturm/10_amiibo_motion.mat')
        load('Dataset/raw/params.mat')
        width = cameraParams.Intrinsics.ImageSize(2);
        height = cameraParams.Intrinsics.ImageSize(1);
        f_gt = mean(cameraParams.Intrinsics.FocalLength);
        Fs = pre_process(Fs);
    case "M3-Amiibo"
        load('Dataset/sturm/12_amiibo_motion.mat')
        load('Dataset/raw/params.mat')
        width = cameraParams.Intrinsics.ImageSize(2);
        height = cameraParams.Intrinsics.ImageSize(1);
        f_gt = mean(cameraParams.Intrinsics.FocalLength);
        Fs = pre_process(Fs);
end

init_focal_length(Fs, width, height, f_gt, xaxis_limit);


function [f0] = init_focal_length(Fs, width, height, f_gt, xaxis_limit)

% Output from methods
disp("Robust initialization")

[mu_1, sigma_1] = sturm1(Fs, width, height);
disp("Sturm (vanilla)")
fprintf("mu = %f, sigma = %f\n", mu_1, sigma_1)

[mu_1_1, sigma_1_1] = sturm1_1(Fs, width, height);
disp("Sturm w/ kernel voting")
fprintf("mu = %f, sigma = %f\n", mu_1_1, sigma_1_1)

disp("Ours")
[mu_f0, sigma_f0, x, ySix] = sturm2(Fs, width, height);
fprintf("mu = %f, sigma = %f\n", mu_f0, sigma_f0)

% Plot KDE
figure('Name', 'Sturm plot results', 'NumberTitle', 'off');
hold on
h = plot((x - f_gt) / f_gt, ySix ./ max(ySix),'k-','LineWidth',3);
set(h,'LineSmoothing','On')
set(gca,'FontSize',26);
xlabel('Relative Error (%)', 'FontSize', 32);
ylabel('Density', 'FontSize', 32);
xticks([-1 0 1]);
yticks([]);
if xaxis_limit
    xlim([-1 1])
end
xline((mu_1_1 - f_gt) / f_gt,'r-','LineWidth',5);
xline((mu_1 - f_gt) / f_gt,'g-','LineWidth',5);
xline((mu_f0 - f_gt) / f_gt,'b-','LineWidth',5);
xline(0,'k--','LineWidth',2)
hold off

end


function Fo = pre_process(Fs)
Fo = [];
num_cameras = size(Fs,3);
for i = 1:num_cameras-1
    for j = i+1:num_cameras
        Fo(:,:,size(Fo,3)+1) = Fs(:,:,i,j,1) / norm(Fs(:,:,i,j,1));
    end
end
Fo(:,:,1) = [];
end