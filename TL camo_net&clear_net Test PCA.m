%% Exp_camo_net & exp_clear_net Test Image PCA
% Showing the feature spaces of camo and clear testing images on the experienced
% networks. This demonstrates the transformations caused by transfer learning.

%% Dimensionality Reduction and Feature Mapping

% Loading previously trained networks
load('exp_camo_net.mat')
load('exp_clear_net.mat')

% Loading Augmented Testing Image Datastores
load('TL_CamoTestds.mat')
load('TL_ClearTestds.mat')

% Extracting activations from FC (23rd) Layer
layer = 'fc';
exp_Camo_netCamo_Test = activations(exp_camo_net,TL_CamoTestds,layer,'OutputAs','rows');
exp_Camo_netClear_Test = activations(exp_camo_net,TL_ClearTestds,layer,'OutputAs','rows');
exp_Clear_netClear_Test = activations(exp_clear_net,TL_ClearTestds,layer,'OutputAs','rows');
exp_Clear_netCamo_Test = activations(exp_clear_net,TL_CamoTestds,layer,'OutputAs','rows');

% Reducing Dimensionality
[coeff1,score1] = pca(exp_Camo_netCamo_Test); 
[coeff2,score2] = pca(exp_Camo_netClear_Test); 
[coeff3,score3] = pca(exp_Clear_netClear_Test);
[coeff4,score4] = pca(exp_Clear_netCamo_Test);
exp_camo_camo_acts = score1(:,1:3); % exp_Camo_net camo image activations
exp_camo_clear_acts = score2(:,1:3); % exp_Camo_net clear image activations
exp_clear_clear_acts = score3(:,1:3); % exp_Clear_net clear image activations
exp_clear_camo_acts = score4(:,1:3); % exp_Clear_net camo image activations

%% Setting Parameters for the For Loops

frog_color = "#018441"; %[0.0050 0.516 0.253]; % #018441
horse_color = "#D95319"; %[0.8500 0.3250 0.0980]; %#D95319
octopus_color = "#4DBEEE"; %[0.3010 0.7450 0.9330]; %#4DBEEE
owl_color = "#A2142F"; %[0.6350 0.0780 0.1840]; %#A2142F
reptile_color = "#77AC30"; %[0.4660 0.6740 0.1880]; %#77AC30
fish_color = "#7E2F8E"; %[0.4940 0.1840 0.5560]; %#7E2F8E
mammal_color = "#EDB120"; %[0.9290 0.6940 0.1250]; %#EDB120
insect_color = "#0072BD"; %[0 0.4470 0.7410]; %#0072BD
def_colors = ["b","r","g","y","c","m","k"];
rgb_colors = [frog_color,horse_color,octopus_color,owl_color,reptile_color,fish_color,mammal_color,insect_color];
colors = [def_colors, rgb_colors];
TL_ClTe_1 = [01,29,52,66,105,137,147,162,196,239,282,308,327,339,358];
TL_ClTe_2 = [28,51,65,104,136,146,161,195,238,281,307,326,338,357,387];
TL_CaTe_1 = [1,17,58,102,121,153,203,245,272,289,306,341,381,402,444];
TL_CaTe_2 = [16,57,101,120,152,202,244,271,288,305,340,380,401,443,472];
d1 = [1 1 2];
d2 = [2 3 3];

%% Mapping the Testing Image Clusters

% exp_Camo_net camo test image FS
figure; 
hold on
for i = 1:length(TL_CaTe_1)
    plot3([exp_camo_camo_acts(TL_CaTe_1(i):TL_CaTe_2(i),1)],[exp_camo_camo_acts(TL_CaTe_1(i):TL_CaTe_2(i),2)],[exp_camo_camo_acts(TL_CaTe_1(i):TL_CaTe_2(i),3)],'*','Color',colors(i))
end
title('ExpCamoNet Camo Test Activations (X-Y)')
legend('Bear','Bird','Bulky Insect','Canine','Feline','Flat Fish','Flat Insect','Frog','Horse Type',...
'Octopus','Owl','Reptile','Small Fish','Small Mammal','Stick Insect','Location','best')

% exp_Camo_net clear test image FS
figure; 
hold on
for i = 1:length(TL_ClTe_1)
    plot3([exp_camo_clear_acts(TL_ClTe_1(i):TL_ClTe_2(i),1)],[exp_camo_clear_acts(TL_ClTe_1(i):TL_ClTe_2(i),2)],[exp_camo_clear_acts(TL_ClTe_1(i):TL_ClTe_2(i),3)],'o','Color',colors(i))
end
title('ExpCamoNet Clear Test Activations (X-Y)')
legend('Bear','Bird','Bulky Insect','Canine','Feline','Flat Fish','Flat Insect','Frog','Horse Type',...
'Octopus','Owl','Reptile','Small Fish','Small Mammal','Stick Insect','Location','best')

% exp_Clear_net clear test image FS
figure; 
hold on
for i = 1:length(TL_ClTe_1)
    plot3([exp_clear_clear_acts(TL_ClTe_1(i):TL_ClTe_2(i),1)],[exp_clear_clear_acts(TL_ClTe_1(i):TL_ClTe_2(i),2)],[exp_clear_clear_acts(TL_ClTe_1(i):TL_ClTe_2(i),3)],'o','Color',colors(i))
end
title('ExpClearNet Clear Test Activations (X-Y)')
legend('Bear','Bird','Bulky Insect','Canine','Feline','Flat Fish','Flat Insect','Frog','Horse Type',...
'Octopus','Owl','Reptile','Small Fish','Small Mammal','Stick Insect','Location','best')

% exp_Clear_net camo test image FS
figure; 
hold on
for i = 1:length(TL_CaTe_1)
    plot3([exp_clear_camo_acts(TL_CaTe_1(i):TL_CaTe_2(i),1)],[exp_clear_camo_acts(TL_CaTe_1(i):TL_CaTe_2(i),2)],[exp_clear_camo_acts(TL_CaTe_1(i):TL_CaTe_2(i),3)],'o','Color',colors(i))
end
title('ExpClearNet Camo Test Activations (X-Y)')
legend('Bear','Bird','Bulky Insect','Canine','Feline','Flat Fish','Flat Insect','Frog','Horse Type',...
'Octopus','Owl','Reptile','Small Fish','Small Mammal','Stick Insect','Location','best')


%% ConvexHull Representation of Previous Figures

% exp_Camo_net camo test image FS
figs = length(findobj('Type','Figure'));
figure;
figure;
figure;
for i = 1:15
    for j = 1:length(d1)
        cluster = double(exp_camo_camo_acts(TL_CaTe_1(i):TL_CaTe_2(i),[d1(j) d2(j)]));  
        k = convhull(cluster);                                              
        figure(j+figs)
        hold on
        plot(cluster(k,1),cluster(k,2),'--','Color',colors(i))              
    end
end

% exp_Camo_net clear test image FS
figs = length(findobj('Type','Figure'));
figure;
figure;
figure;
for i = 1:15
    for j = 1:length(d1)
        cluster = double(exp_camo_clear_acts(TL_ClTe_1(i):TL_ClTe_2(i),[d1(j) d2(j)]));  
        k = convhull(cluster);                                              
        figure(j+figs)
        hold on
        plot(cluster(k,1),cluster(k,2),'--','Color',colors(i))              
    end
end

% exp_Clear_net clear test image FS
figs = length(findobj('Type','Figure'));
figure;
figure;
figure;
for i = 1:15
    for j = 1:length(d1)
        cluster = double(exp_clear_clear_acts(TL_ClTe_1(i):TL_ClTe_2(i),[d1(j) d2(j)]));  
        k = convhull(cluster);                                              
        figure(j+figs)
        hold on
        plot(cluster(k,1),cluster(k,2),'--','Color',colors(i))              
    end
end

% exp_Clear_net camo test image FS
figs = length(findobj('Type','Figure'));
figure;
figure;
figure;
for i = 1:15
    for j = 1:length(d1)
        cluster = double(exp_clear_camo_acts(TL_CaTe_1(i):TL_CaTe_2(i),[d1(j) d2(j)]));  
        k = convhull(cluster);                                              
        figure(j+figs)
        hold on
        plot(cluster(k,1),cluster(k,2),'--','Color',colors(i))              
    end
end


