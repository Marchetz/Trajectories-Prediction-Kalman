framepath = '/media/becattini/RedPro/dataset/kitti_raw_data/2011_09_26/2011_09_26_drive_0001_sync/image_02/data/';
allfiles = arrayfun(@(x)x.name, dir('data/*.txt'), 'un', 0);

fx = @(x,y)cos(y).*cos(x);
fy = @(x,y)sin(y).*cos(x);
fz = @(x,y)sin(x);

for i = 1:length(allfiles)
    data(i,:) = textread(['./data/' allfiles{i}]);
end

figure;
xdata = fx(data(:,1), data(:,2));
ydata = -fy(data(:,1), data(:,2));
zdata = fz(data(:,1), data(:,2));

A = [0 0 0];
B = [2 0 0];
C = [0 1 0];
D = [0 0 1];
E = [0 1 0.5];
F = [2 0 1];
G = [1 1 0];
H = [1 1 0.5];
P0 = [A;B;F;H;G;C;A;D;E;H;F;D;E;C;G;B];

for i = 1:length(data)
    
%     lat = data(i, 1);
%     lon = data(i, 2);
%     n = dcmecef2ned(lat, lon);
    
    subplot(2,2,[1 2]);
    imshow(imread([framepath '/' strrep(allfiles{i}, 'txt', 'png')]));
    
    subplot(2,2,3);
    plot(xdata(1:i), ydata(1:i));
    xlim([min(xdata), max(xdata)]);
    ylim([min(ydata), max(ydata)]);
    
    subplot(2,2,4);
    roll = data(i, 4);
    pitch = data(i, 5);
    yaw = data(i, 6);
    dcm = angle2dcm(yaw, pitch, roll);
    P = P0*dcm;
    P(:,1) = P(:,1);
    P(:,2) = P(:,2);
    P(:,3) = P(:,3);
    plot3(P(:,1),P(:,2),P(:,3)); % rotated cube
%     hold on;
    xlim([-2 1]); ylim([-2 1]); zlim([-2 1]);
    drawnow;
end