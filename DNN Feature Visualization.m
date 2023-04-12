%% Visualization of DNN features

% CamoNet Visualization
load('camo_net.mat')

analyzeNetwork(camo_net) % Conv layers are 2,6, 10, 12, 14

layer = 2;
name = camo_net.Layers(layer).Name

channels = 1:36; % Visualize the first 36 features learned by this layer using deepDreamImage by setting channels to be the vector of indices 1:36. Set 'PyramidLevels' to 1 so that the images are not scaled. To display the images together, you can use imtile.
I = deepDreamImage(camo_net,name,channels,'PyramidLevels',1);

figure
I = imtile(I,'ThumbnailSize',[64 64]);
imshow(I)
title(['Layer ',name,' Features'],'Interpreter','none') % Layer 2 (Conv 1) features

layer = 6;
name = camo_net.Layers(layer).Name

I = deepDreamImage(camo_net,name,channels,'Verbose',false,'PyramidLevels',1);
figure
I = imtile(I,'ThumbnailSize',[64 64]);
imshow(I)
title(['Layer ',name,' Features'],'Interpreter','none') % Layer 6 (Conv 2) features

layer = 10
name = camo_net.Layers(layer).Name
channels = 1:6;
I = deepDreamImage(camo_net,name,channels,'Verbose',false,'NumIterations',20,'PyramidLevels',2);
figure
I = imtile(I,'ThumbnailSize',[250 250]);
imshow(I)
name = camo_net.Layers(layer).Name;
title(['Layer ',name,' Features'],'Interpreter','none')


layer = 12;
name = camo_net.Layers(layer).Name
channels = 1:6;
I = deepDreamImage(camo_net,name,channels,'Verbose',false,'NumIterations',20,'PyramidLevels',2);
figure
I = imtile(I,'ThumbnailSize',[250 250]);
imshow(I)
name = camo_net.Layers(layer).Name;
title(['Layer ',name,' Features'],'Interpreter','none')


layer = 14;
name = camo_net.Layers(layer).Name
channels = 1:6;
I = deepDreamImage(camo_net,name,channels,'Verbose',false,'NumIterations',20,'PyramidLevels',2);
figure
I = imtile(I,'ThumbnailSize',[250 250]);
imshow(I)
name = camo_net.Layers(layer).Name;
title(['Layer ',name,' Features'],'Interpreter','none')



% Visualize Fully Connected Layer
layer = 23;
name = camo_net.Layers(layer).Name

channels = [8 12 3 4 11 6];
camo_net.Layers(end).Classes(channels)

I = deepDreamImage(camo_net,name,channels,'Verbose',false,'NumIterations',100,'PyramidLevels',2);
figure
I = imtile(I,'ThumbnailSize',[250 250]);
imshow(I)
name = net.Layers(layer).Name;
title(['Layer ',name,' Features'])

% ClearNet
load('clear_net.mat')

analyzeNetwork(clear_net) % Conv layers are 2,6, 10, 12, 14

layer = 2;
name = clear_net.Layers(layer).Name

channels = 1:36; % Visualize the first 36 features learned by this layer using deepDreamImage by setting channels to be the vector of indices 1:36. Set 'PyramidLevels' to 1 so that the images are not scaled. To display the images together, you can use imtile.
I = deepDreamImage(clear_net,name,channels,'PyramidLevels',1);

figure
I = imtile(I,'ThumbnailSize',[64 64]);
imshow(I)
title(['Layer ',name,' Features'],'Interpreter','none') % Layer 2 (Conv 1) features

layer = 6;
name = clear_net.Layers(layer).Name

I = deepDreamImage(clear_net,name,channels,'Verbose',false,'PyramidLevels',1);
figure
I = imtile(I,'ThumbnailSize',[64 64]);
imshow(I)
title(['Layer ',name,' Features'],'Interpreter','none') % Layer 6 (Conv 2) features

layer = 10
name = clear_net.Layers(layer).Name
channels = 1:6;
I = deepDreamImage(clear_net,name,channels,'Verbose',false,'NumIterations',20,'PyramidLevels',2);
figure
I = imtile(I,'ThumbnailSize',[250 250]);
imshow(I)
name = clear_net.Layers(layer).Name;
title(['Layer ',name,' Features'],'Interpreter','none')


layer = 12;
name = clear_net.Layers(layer).Name
channels = 1:6;
I = deepDreamImage(clear_net,name,channels,'Verbose',false,'NumIterations',20,'PyramidLevels',2);
figure
I = imtile(I,'ThumbnailSize',[250 250]);
imshow(I)
name = clear_net.Layers(layer).Name;
title(['Layer ',name,' Features'],'Interpreter','none')


layer = 14;
name = clear_net.Layers(layer).Name
channels = 1:6;
I = deepDreamImage(clear_net,name,channels,'Verbose',false,'NumIterations',20,'PyramidLevels',2);
figure
I = imtile(I,'ThumbnailSize',[250 250]);
imshow(I)
name = clear_net.Layers(layer).Name;
title(['Layer ',name,' Features'],'Interpreter','none')



% Visualize Fully Connected Layer
layer = 23;
name = clear_net.Layers(layer).Name

channels = [1 2 3 4 5 6];
clear_net.Layers(end).Classes(channels)

I = deepDreamImage(clear_net,name,channels,'Verbose',false,'NumIterations',100,'PyramidLevels',2);
figure
I = imtile(I,'ThumbnailSize',[250 250]);
imshow(I)
name = net.Layers(layer).Name;
title(['Layer ',name,' Features'])


%% Feature Space visualization of Layer features

load('camo_net.mat')

load('CamoTestds2.mat')

layer = 'fc';
CamoNetfeaturesTest = activations(camo_net,CamoTestds2,layer,'OutputAs','rows');

FCBearImg = imread('CamoNet FC Bear (cropped).png');
FCBearImg = imresize(FCBearImg,[227 227]);
FCCanineImg = imread('CamoNet FC Canine (cropped).png');
FCCanineImg = imresize(FCCanineImg,[227 227]);
FCFrogImg = imread('CamoNet FC Frog (cropped).png');
FCFrogImg = imresize(FCFrogImg,[227 227]);
FCReptileImg = imread('CamoNet FC Reptile (cropped).png');
FCReptileImg = imresize(FCReptileImg,[227 227]);

FCBearActs = activations(camo_net,FCBearImg,layer,'OutputAs','rows');
FCCanineActs = activations(camo_net,FCCanineImg,layer,'OutputAs','rows');
FCFrogActs = activations(camo_net,FCFrogImg,layer,'OutputAs','rows');
FCReptileActs = activations(camo_net,FCReptileImg,layer,'OutputAs','rows');

CamoNetfeaturesTest = cat(1,CamoNetfeaturesTest,FCBearActs);
CamoNetfeaturesTest = cat(1,CamoNetfeaturesTest,FCCanineActs);
CamoNetfeaturesTest = cat(1,CamoNetfeaturesTest,FCFrogActs);
CamoNetfeaturesTest = cat(1,CamoNetfeaturesTest,FCReptileActs);

% Reducing Dimensionality
[coeff,score] = pca(CamoNetfeaturesTest);
CamoNet_Acts = score(:,1:2);

% Assigning Camo Animal Clusters
CamoBear = CamoNet_Acts(1:30,:);
CamoCan = CamoNet_Acts(143:185,:);
CamoFrog = CamoNet_Acts(352:409,:);
CamoRept = CamoNet_Acts(546:598,:);

% Plotting Camo Animal Group Clusters
figure;
hold on
plot(CamoBear(:,1),CamoBear(:,2),'bd')
plot(CamoCan(:,1),CamoCan(:,2),'rd')
plot(CamoFrog(:,1),CamoFrog(:,2),'gd')
plot(CamoRept(:,1),CamoRept(:,2),'blackd')
xlim([-80 60]);ylim([-15 15])

% Plotting FC Feature Image Points
hold on
figure;
hold on
plot(CamoNet_Acts(721,1),CamoNet_Acts(721,2),'b*')
plot(CamoNet_Acts(722,1),CamoNet_Acts(722,2),'r*')
plot(CamoNet_Acts(723,1),CamoNet_Acts(723,2),'g*')
plot(CamoNet_Acts(724,1),CamoNet_Acts(724,2),'black*')
ylim([-15 15])




%% Trying new FS true centers

load('camo_net.mat')

load('CamoTestds2.mat')


% Extracting activations from FC (23rd) Layer
layer = 'fc';
Camo_NetCamofeaturesTest = activations(camo_net,CamoTestds2,layer,'OutputAs','rows');

Camo_NetCamofeaturesTest = cat(1,Camo_NetCamofeaturesTest,FCBearActs);
Camo_NetCamofeaturesTest = cat(1,Camo_NetCamofeaturesTest,FCCanineActs);
Camo_NetCamofeaturesTest = cat(1,Camo_NetCamofeaturesTest,FCFrogActs);
Camo_NetCamofeaturesTest = cat(1,Camo_NetCamofeaturesTest,FCReptileActs);

% Reducing Dimensionality
[coeff1,score1] = pca(Camo_NetCamofeaturesTest);
camo_net_Camotest_acts = score1(:,1:2);

% Assigning Camo Animal Clusters
CamoBear = camo_net_Camotest_acts(1:30,:);
CamoBear = cat(1,CamoBear,camo_net_Camotest_acts(721,:));
CamoCan = camo_net_Camotest_acts(143:185,:);
CamoCan = cat(1,CamoCan,camo_net_Camotest_acts(722,:));
CamoFrog = camo_net_Camotest_acts(352:409,:);
CamoFrog = cat(1,CamoFrog,camo_net_Camotest_acts(723,:));
CamoRept = camo_net_Camotest_acts(546:598,:);
CamoRept = cat(1,CamoRept,camo_net_Camotest_acts(724,:));

% Plotting Camo Animal Group Clusters
figure;
hold on
plot(CamoBear(:,1),CamoBear(:,2),'bd')
plot(CamoCan(:,1),CamoCan(:,2),'rd')
plot(CamoFrog(:,1),CamoFrog(:,2),'gd')
plot(CamoRept(:,1),CamoRept(:,2),'blackd')
hold on
plot(CamoNet_Acts(721,1),CamoNet_Acts(721,2),'b*')
plot(CamoNet_Acts(722,1),CamoNet_Acts(722,2),'r*')
plot(CamoNet_Acts(723,1),CamoNet_Acts(723,2),'g*')
plot(CamoNet_Acts(724,1),CamoNet_Acts(724,2),'black*')

% Calculating Camo Cluster Centroids
BearMean = [mean(CamoBear(:,1)),mean(CamoBear(:,2))];
CanMean = [mean(CamoCan(:,1)),mean(CamoCan(:,2))];
FrogMean = [mean(CamoFrog(:,1)),mean(CamoFrog(:,2))];
ReptMean = [mean(CamoRept(:,1)),mean(CamoRept(:,2))];

% Calculating Centroid Distance (Bear, Camo Testing)
SubtBear = CamoBear - BearMean;
SqrBear = SubtBear .^ 2;
SumBear = sum(SqrBear,2);
DistBear = sqrt(SumBear);

% Calculating Centroid Distance (Canine, Camo Testing)
SubtCan = CamoCan - CanMean;
SqrCan = SubtCan .^ 2;
SumCan = sum(SqrCan,2);
DistCan = sqrt(SumCan);

% Calculating Centroid Distance (Frog, Camo Testing)
SubtFrog = CamoFrog - FrogMean;
SqrFrog = SubtFrog .^ 2;
SumFrog = sum(SqrFrog,2);
DistFrog = sqrt(SumFrog);

% Calculating Centroid Distance (Reptile, Camo Testing)
SubtRept = CamoRept - ReptMean;
SqrRept = SubtRept .^ 2;
SumRept = sum(SqrRept,2);
DistRept = sqrt(SumRept);


CamoBearActs = Camo_NetCamofeaturesTest(1:30,1);
CamoBearActs = cat(1,CamoBearActs,Camo_NetCamofeaturesTest(721,1));

% Camo Bear True Center  
for x = -30:1:-25 %borders of cluster on x-axis 
    for y = 3:1:5 %borders of cluster on y-axis
        SubtBear = (CamoBear(:,:)) - [-26.5 2.3];
        SqrBear = SubtBear .^ 2;
        SumBear = sum(SqrBear,2);
        DistBear = sqrt(SumBear);
        figure;
        %plot(DistBear,CamoBear(:,1),'b*')
        plot(DistBear,CamoBearActs,'b*')
        title(['x= ' num2str(x) 'y= ' num2str(y)])
    end
end
%(-26,4) - exponential
%(-30,3) - linear
% Read about feature map generation -> make the case that Deep Dream does
% in fact generate a prototype (for meeting w/ Haluk)

% Note that FS prototype points are along a diagonal line. The BC/FR
% prototype points are maximally distances along this line, which explains
% why they're not at the centroid. This line is almost parallel w/ 1st PC,
% minimally distributed along Y-axis

% Add to Gdrive

CamoCanActs = Camo_NetCamofeaturesTest(143:185,4);
CamoCanActs = cat(1,CamoCanActs,Camo_NetCamofeaturesTest(722,4));

% Camo Canine True Center
for x = -25:1:-23 %borders of cluster on x-axis
    for y = 2:1:4 %borders of cluster on y-axis
        SubtCan = (CamoCan(:,:)) - [x y];
        SqrCan = SubtCan .^ 2;
        SumCan = sum(SqrCan,2);
        DistCan = sqrt(SumCan);
        figure;
        %plot(DistCan,CamoCan(:,1),'r*')
        plot(DistCan,CamoCanActs,'r*')
        title(['x= ' num2str(x) 'y= ' num2str(y)])
    end
end
%(-24,3)

CamoFrogActs = Camo_NetCamofeaturesTest(352:409,8);
CamoFrogActs = cat(1,CamoFrogActs,Camo_NetCamofeaturesTest(723,8));

% Camo Frog
for x = 20:1:22 %borders of cluster on x-axis
    for y = 5:1:10 %borders of cluster on y-axis
        SubtFrog = (CamoFrog(:,:)) - [x y];
        SqrFrog = SubtFrog .^ 2;
        SumFrog = sum(SqrFrog,2);
        DistFrog = sqrt(SumFrog);
        figure;
        %plot(DistFrog,CamoFrog(:,1),'g*')
        plot(DistFrog,CamoFrogActs,'g*')
        title(['x= ' num2str(x) 'y= ' num2str(y)])
    end
end
%(20,5)

CamoReptActs = Camo_NetCamofeaturesTest(546:598,12);
CamoReptActs = cat(1,CamoReptActs,Camo_NetCamofeaturesTest(724,12));

% Camo Reptile True Center
for x = 30:1:34
    for y = -5:1:0
        SubtRept = (CamoRept(:,:)) - [x y];
        SqrRept = SubtRept .^ 2;
        SumRept = sum(SqrRept,2);
        DistRept = sqrt(SumRept);
        figure;
        %plot(DistRept,CamoRept(:,1),'black*')
        plot(DistRept,CamoReptActs,'black*')
        title(['x= ' num2str(x) 'y= ' num2str(y)])
    end
end
%(32,-4)


