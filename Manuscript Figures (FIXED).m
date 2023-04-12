%% Manuscript Figures (FIXED)
% Remaking the manuscript figures that features the error regarding the use
% of correct activations for generalization plots

%% Fig 10 - CamoNet BCFR AvD Plots (REDO w/ PCA)

% Loading previously trained networks and datasets
load('camo_net.mat')
load('CamoTestds2.mat')

% Extracting activations from FC (23rd) Layer
layer = 'fc';
Camo_netCamoActs = activations(camo_net,CamoTestds2,layer,'OutputAs','rows');

% Extracting and Combining BCFR matrix (CamoNet)
CamoBear = Camo_netCamoActs(1:30,:);
CamoCanine = Camo_netCamoActs(143:185,:);
CamoFrog = Camo_netCamoActs(352:409,:);
CamoReptile = Camo_netCamoActs(546:598,:);
CombAnimals = [CamoBear;CamoCanine;CamoFrog;CamoReptile];

% Euclidean Distance Calculation for BCFR MDS (CamoNet)
Dist1 = NaN(184,184);
for i = 1:184
    for j = 1:184
        Dist1(i,j) = sqrt(sum((CombAnimals(i,:)- CombAnimals(j,:)).^2,2));
    end
end

% MDS BCFR (CamoNet)
CamoNetMDS_BCFR = mdscale(Dist1,3);

% Plotting BCFR Clusters (ExpCamoNet)
figure;
hold on
plot3(CamoNetMDS_BCFR(1:30,1),CamoNetMDS_BCFR(1:30,2),CamoNetMDS_BCFR(1:30,3),'b*'); % Bear
plot3(CamoNetMDS_BCFR(31:73,1),CamoNetMDS_BCFR(31:73,2),CamoNetMDS_BCFR(31:73,3),'r*'); % Canine
plot3(CamoNetMDS_BCFR(74:131,1),CamoNetMDS_BCFR(74:131,2),CamoNetMDS_BCFR(74:131,3),'g*') % Frog       
plot3(CamoNetMDS_BCFR(132:184,1),CamoNetMDS_BCFR(132:184,2),CamoNetMDS_BCFR(132:184,3),'black*') % Reptile

% Establishing Camo BCFR Animals Clusters (Post MDS, ExpCamoNet)
CamoBearMDS = CamoNetMDS_BCFR(1:30,1:2);
CamoCanineMDS = CamoNetMDS_BCFR(31:73,1:2);
CamoFrogMDS = CamoNetMDS_BCFR(74:131,1:2);
CamoReptileMDS = CamoNetMDS_BCFR(132:184,1:2);

% Extracting Animal-specific Activations (from FC layer)
CamoBearActs = Camo_netCamoActs(1:30,1);
CamoCanineActs = Camo_netCamoActs(143:185,4);
CamoFrogActs = Camo_netCamoActs(352:409,8);
CamoReptileActs = Camo_netCamoActs(546:598,12);

% CamoNet Bear True Center
for x = 20:2:30 %borders of cluster on x-axis 
    for y = -10:2:-6 %borders of cluster on y-axis
        SubtBear = (CamoBearMDS(:,:)) - [30 -8];
        SqrBear = SubtBear .^ 2;
        SumBear = sum(SqrBear,2);
        DistBear = sqrt(SumBear);
        figure;
        plot(DistBear,CamoBearActs,'b*')
        title(['x= ' num2str(x) 'y= ' num2str(y)])
    end
end
% (30,-8)

% CamoNet Canine True Center
for x = 23:1:27 %borders of cluster on x-axis 
    for y = 5:1:10 %borders of cluster on y-axis
        SubtCan = (CamoCanineMDS(:,:)) - [25 8];
        SqrCan = SubtCan.^ 2;
        SumCan = sum(SqrCan,2);
        DistCan = sqrt(SumCan);
        figure;
        plot(DistCan,CamoCanineActs,'r*')
        title(['x= ' num2str(x) 'y= ' num2str(y)])
    end
end
%(25,8)

% CamoNet Frog True Center
for x = -20:1:-15 %borders of cluster on x-axis 
    for y = 7:1:13 %borders of cluster on y-axis
        SubtFrog = (CamoFrogMDS(:,:)) - [-20 10];
        SqrFrog = SubtFrog.^ 2;
        SumFrog = sum(SqrFrog,2);
        DistFrog = sqrt(SumFrog);
        figure;
        plot(DistFrog,CamoFrogActs,'g*')
        title(['x= ' num2str(x) 'y= ' num2str(y)])
    end
end
%(-20,10)

% CamoNet Reptile True Center
for x = -30:1:-25 %borders of cluster on x-axis 
    for y = -15:1:-5 %borders of cluster on y-axis
        SubtRept = (CamoReptileMDS(:,:)) - [-25 -10];
        SqrRept = SubtRept.^ 2;
        SumRept = sum(SqrRept,2);
        DistRept = sqrt(SumRept);
        figure;
        plot(DistRept,CamoReptileActs,'black*')
        title(['x= ' num2str(x) 'y= ' num2str(y)])
    end
end
%(-25,-10)

%% Figure 11 - Manhattan Curves

% Loading previously trained networks and datasets
load('exp_camo_net.mat')
load('CamoTestds2.mat')

% Extracting activations from FC (23rd) Layer
layer = 'fc';
ExpCamo_netCamoActs = activations(exp_camo_net,CamoTestds2,layer,'OutputAs','rows');

% Extracting and Combining BCFR matrix (ExpCamoNet)
CamoBear = ExpCamo_netCamoActs(1:30,:);
CamoCanine = ExpCamo_netCamoActs(143:185,:);
CamoFrog = ExpCamo_netCamoActs(352:409,:);
CamoReptile = ExpCamo_netCamoActs(546:598,:);
CombAnimals = [CamoBear;CamoCanine;CamoFrog;CamoReptile];

% Manhattan Distance Calculation for BCFR MDS (ExpCamoNet)
Dist1 = NaN(184,184);
for i = 1:184
    for j = 1:184
        Dist1(i,j) = sum(abs(CombAnimals(i,:)- CombAnimals(j,:)),2);
    end
end

% MDS BCFR (ExpCamoNet)
ExpCamoNetMDS_BCFR = mdscale(Dist1,3); % Manhattan Distance MDS

% Plotting BCFR Clusters (ExpCamoNet)
figure;
hold on
plot3(ExpCamoNetMDS_BCFR(1:30,1),ExpCamoNetMDS_BCFR(1:30,2),ExpCamoNetMDS_BCFR(1:30,3),'b*'); % Bear
plot3(ExpCamoNetMDS_BCFR(31:73,1),ExpCamoNetMDS_BCFR(31:73,2),ExpCamoNetMDS_BCFR(31:73,3),'r*'); % Canine
plot3(ExpCamoNetMDS_BCFR(74:131,1),ExpCamoNetMDS_BCFR(74:131,2),ExpCamoNetMDS_BCFR(74:131,3),'g*') % Frog       
plot3(ExpCamoNetMDS_BCFR(132:184,1),ExpCamoNetMDS_BCFR(132:184,2),ExpCamoNetMDS_BCFR(132:184,3),'black*') % Reptile    

% Establishing Camo BCFR Animals Clusters (Post MDS, ExpCamoNet)
ExpCamoBearMDS = ExpCamoNetMDS_BCFR(1:30,1:2);
ExpCamoCanineMDS = ExpCamoNetMDS_BCFR(31:73,1:2);
ExpCamoFrogMDS = ExpCamoNetMDS_BCFR(74:131,1:2);
ExpCamoReptileMDS = ExpCamoNetMDS_BCFR(132:184,1:2);

% Extracting Animal-specific Activations (from FC layer)
ExpCamoBearActs = ExpCamo_netCamoActs(1:30,1);
ExpCamoCanineActs = ExpCamo_netCamoActs(143:185,4);
ExpCamoFrogActs = ExpCamo_netCamoActs(352:409,8);
ExpCamoReptileActs = ExpCamo_netCamoActs(546:598,12);

% ExpCamoNet Bear True Center
for x = -60:20:80 %borders of cluster on x-axis 
    for y = -40:30:50 %borders of cluster on y-axis
        SubtBear = (ExpCamoBearMDS(:,:)) - [58 3];
        SqrBear = SubtBear .^ 2;
        SumBear = sum(SqrBear,2);
        DistBear = sqrt(SumBear);
        figure;
        plot(DistBear,ExpCamoBearActs(:,1),'b*')
        title(['x= ' num2str(x) 'y= ' num2str(y)])
    end
end
% (58,3)

% ExpCamoNet Canine True Center
for x = 55:2:65 %borders of cluster on x-axis
    for y = -5:2:5 %borders of cluster on y-axis
        SubtCan = (ExpCamoCanineMDS(:,:)) - [61 -5];
        SqrCan = SubtCan .^ 2;
        SumCan = sum(SqrCan,2);
        DistCan = sqrt(SumCan);
        figure;
        plot(DistCan,ExpCamoCanineActs(:,1),'r*')
        title(['x= ' num2str(x) 'y= ' num2str(y)])
    end
end
% (61,-5)

% Camo Frog True Center (ExpCamoNet)
for x = -50:2:-40 %borders of cluster on x-axis
    for y = -40:2:-20 %borders of cluster on y-axis
        SubtFrog = (ExpCamoFrogMDS(:,:)) - [-48 -28];
        SqrFrog = SubtFrog .^ 2;
        SumFrog = sum(SqrFrog,2);
        DistFrog = sqrt(SumFrog);
        figure;
        plot(DistFrog,ExpCamoFrogActs(:,1),'g*')
        title(['x= ' num2str(x) 'y= ' num2str(y)])
    end
end
% (-48,-28)

% Camo Reptile True Center (ExpCamoNet)
for x = -50:2:-40
    for y = 5:2:15
        SubtRept = (ExpCamoReptileMDS(:,:)) - [-42 9];
        SqrRept = SubtRept .^ 2;
        SumRept = sum(SqrRept,2);
        DistRept = sqrt(SumRept);
        figure;
        plot(DistRept,ExpCamoReptileActs(:,1),'black*')
        title(['x= ' num2str(x) 'y= ' num2str(y)])
    end
end
% (-42,9)

%% Figure 11 - PCA Curves

% Loading previously trained networks and datasets
load('exp_camo_net.mat')
load('CamoTestds2.mat')

% Extracting activations from FC (23rd) Layer
layer = 'fc';
ExpCamoNetCamoActs = activations(exp_camo_net,CamoTestds2,layer,'OutputAs','rows');

% Reducing Dimensionality
[coeff1,score1] = pca(ExpCamoNetCamoActs);
ExpCamoNet_PCA = score1(:,1:2);

% Assigning Camo Animal Activations
ExpCamoBearActs = ExpCamoNetCamoActs(1:30,1);
ExpCamoCanineActs = ExpCamoNetCamoActs(143:185,4);
ExpCamoFrogActs = ExpCamoNetCamoActs(352:409,8);
ExpCamoReptileActs = ExpCamoNetCamoActs(546:598,12);

% Assigning Camo Animal Clusters
CamoBear = ExpCamoNet_PCA(1:30,:);
CamoCan = ExpCamoNet_PCA(143:185,:);
CamoFrog = ExpCamoNet_PCA(352:409,:);
CamoRept = ExpCamoNet_PCA(546:598,:);

% Plotting Camo Animal Group Clusters
figure;
hold on
plot(CamoBear(:,1),CamoBear(:,2),'b*')
plot(CamoCan(:,1),CamoCan(:,2),'r*')
plot(CamoFrog(:,1),CamoFrog(:,2),'g*')
plot(CamoRept(:,1),CamoRept(:,2),'black*')
title('ExpCamoNet Camo BCFR Test Activations')

% Camo Bear True Center  
for x = 18:1:22 %borders of cluster on x-axis 
    for y = 7:1:13 %borders of cluster on y-axis
        SubtBear = (CamoBear(:,:)) - [22 7];
        SqrBear = SubtBear .^ 2;
        SumBear = sum(SqrBear,2);
        DistBear = sqrt(SumBear);
        figure;
        plot(DistBear,ExpCamoBearActs(:,1),'b*')
        title(['x= ' num2str(x) 'y= ' num2str(y)])
    end
end
% (22,7) is the true center

% Camo Canine True Center
for x = 18:1:22 %borders of cluster on x-axis
    for y = 5:1:9 %borders of cluster on y-axis
        SubtCan = (CamoCan(:,:)) - [20 7];
        SqrCan = SubtCan .^ 2;
        SumCan = sum(SqrCan,2);
        DistCan = sqrt(SumCan);
        %figure;
        plot(DistCan,ExpCamoCanineActs(:,1),'r*')
        title(['x= ' num2str(x) 'y= ' num2str(y)])
    end
end
% (20,7) is the true center

% Camo Frog True Center
for x = -17:1:-11 %borders of cluster on x-axis
    for y = -3:1:1 %borders of cluster on y-axis
        SubtFrog = (CamoFrog(:,:)) - [-11 -3];
        SqrFrog = SubtFrog .^ 2;
        SumFrog = sum(SqrFrog,2);
        DistFrog = sqrt(SumFrog);
        figure;
        plot(DistFrog,ExpCamoFrogActs(:,1),'g*')
        title(['x= ' num2str(x) 'y= ' num2str(y)])
    end
end
% (-11,-3) is the true center

% Camo Reptile True Center
for x = -15:1:-10
    for y = -15:1:-10
        SubtRept = (CamoRept(:,:)) - [-14 -11];
        SqrRept = SubtRept .^ 2;
        SumRept = sum(SqrRept,2);
        DistRept = sqrt(SumRept);
        figure;
        plot(DistRept,ExpCamoReptileActs(:,1),'black*')
        title(['x= ' num2str(x) 'y= ' num2str(y)])
    end
end
% (-14,-11) is the true center















