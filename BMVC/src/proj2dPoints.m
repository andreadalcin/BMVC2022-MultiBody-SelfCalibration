function UV_k = proj2dPoints(XYZ_k, K, R, t)
num_points = size(XYZ_k,2);
num_objects = size(XYZ_k,3);

% Parameters
t_std = 1e0;        % Translation standard deviation
min_angle = 0;      % Min rotation angle
max_angle = 365;    % Max rotation angle

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Generate rotations
Rx_k = zeros(3,3,num_objects);
Ry_k = zeros(3,3,num_objects);
Rz_k = zeros(3,3,num_objects);

for i = 1:num_objects
    Rx_k(:,:,i) = rotx(min_angle + (max_angle - min_angle).*rand(1,1));
    Ry_k(:,:,i) = roty(min_angle + (max_angle - min_angle).*rand(1,1));
    Rz_k(:,:,i) = rotz(min_angle + (max_angle - min_angle).*rand(1,1));
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Move 3D objects and plot
for i = 1:num_objects
    R_i = Rz_k(:,:,i) * Ry_k(:,:,i) * Rz_k(:,:,i);
    t_i = t_std * rand(3,1);

    XYZ_k(:,:,i) = R_i * XYZ_k(:,:,i) + t_i;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
% Project points to 2D image
% plot3D(XYZ_k, R, t);
P = K * [1, 0, 0, 0; 0, 1, 0, 0; 0, 0, 1, 0] * [R, t'; 0, 0, 0, 1];

UV_k = zeros(2,num_points,num_objects);

% figure
for i = 1:num_objects
    XYZ = XYZ_k(:,:,i);
    XYZ = [XYZ; ones(1,size(XYZ,2))];

    UV = P * XYZ;
    UV = UV ./ UV(3,:);
    UV_k(:,:,i) = UV(1:2,:);
    
    % plot(UV(1,:), UV(2,:), '.')
    % hold on
end
% axis equal
% axis([-1 1 -1 1])
end


function plot3D(XYZ_k, R, t)
k = size(XYZ_k,3);
X_k = XYZ_k(1,:,:);
Y_k = XYZ_k(2,:,:);
Z_k = XYZ_k(3,:,:);

colorstring = 'bgry';
figure
for i = 1:k
    plot3(X_k(:,:,i), Y_k(:,:,i), Z_k(:,:,i), '.','markersize', 1, 'Color', colorstring(i));
    hold on
end

pose = rigid3d(R,t);
plotCamera('AbsolutePose', pose, 'Opacity', 0, 'Size', 0.1);

hold off
axis equal vis3d
end
