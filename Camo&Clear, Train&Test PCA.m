%% Camo and Clear, Training and Testing PCA
% Mapping the FS for camo and clear training and testing images. These two networks are 
% individually trained (no transfer learning). The clusters of the different activations
% are compared to see how the test images cluster.

%% Dimensionality Reduction and Feature Mapping

% Loading previously trained networks
load('camo_net2.mat')
load('clear_net2.mat')

% Loading Augmented Image Datastores used for training networks
load('CamoTrainds2.mat')
load('CamoTestds2.mat')
load('ClearTrainds2.mat')
load('ClearTestds2.mat')

% Extracting activations from FC (23rd) Layer
layer = 'fc';
CamofeaturesTrain = activations(camo_net2,CamoTrainds2,layer,'OutputAs','rows');
CamofeaturesTest = activations(camo_net2,CamoTestds2,layer,'OutputAs','rows');
ClearfeaturesTrain = activations(clear_net2,ClearTrainds2,layer,'OutputAs','rows');
ClearfeaturesTest = activations(clear_net2,ClearTestds2,layer,'OutputAs','rows');

% Reducing Dimensionality
[coeff1,score1] = pca(CamofeaturesTrain); 
[coeff2,score2] = pca(CamofeaturesTest); 
[coeff3,score3] = pca(ClearfeaturesTrain);
[coeff4,score4] = pca(ClearfeaturesTest);
camo_train_acts = score1(:,1:3);
camo_test_acts = score2(:,1:3);
clear_train_acts = score3(:,1:3);
clear_test_acts = score4(:,1:3);


%% Setting Parameters for the For Loops

frog_color = "#018441"; %[0.0050 0.516 0.253]; % #018441
horse_color = "#D95319"; %[0.8500 0.3250 0.0980]; %#D95319
octopus_color = "#4DBEEE"; %[0.3010 0.7450 0.9330]; %#4DBEEE
owl_color = "#A2142F"; %[0.6350 0.0780 0.1840]; %#A2142F
reptile_color = "#77AC30"; %[0.4660 0.6740 0.1880]; %#77AC30
fish_color = "#7E2F8E"; %[0.4940 0.1840 0.5560]; %#7E2F8E
mammal_color = "#EDB120"; %[0.9290 0.6940 0.1250]; %#EDB120
insect_color = "#0072BD"; %[0 0.4470 0.7410]; %#0072BD
def_colors = ['b','r','g','y','c','m','k'];
rgb_colors = [frog_color,horse_color,octopus_color,owl_color,reptile_color,fish_color,mammal_color,insect_color];
ClTr_1 = [1,57,103,131,197,261,280,311,370,434,508,560,598,621,658];
ClTr_2 = [56,102,130,196,260,279,310,369,433,507,559,597,620,657,717];
ClTe_1 = [1,38,68,87,131,173,185,205,245,287,337,371,397,413,438];
ClTe_2 = [37,67,86,130,172,184,204,244,286,336,370,396,412,437,477];
CaTr_1 = [1,45,126,214,278,341,442,527,613,672,747,816,895,936,1020];
CaTr_2 = [44,125,213,277,340,441,526,612,671,746,815,894,935,1019,1077];
CaTe_1 = [1,31,85,143,186,228,296,352,410,450,500,546,599,627,683];
CaTe_2 = [30,84,142,185,227,295,351,409,449,499,545,598,626,682,720];

%% Mapping the Clear Training and Clear Testing Clusters

% Clear Training
figure; 
hold on
for i = 1:length(ClTr_1)
    if i <= 7
        plot3([clear_train_acts(ClTr_1(i):ClTr_2(i),1)],[clear_train_acts(ClTr_1(i):ClTr_2(i),2)],[clear_train_acts(ClTr_1(i):ClTr_2(i),3)],'o','Color',def_colors(i))
    else
        j = i - length(def_colors);
        plot3([clear_train_acts(ClTr_1(i):ClTr_2(i),1)],[clear_train_acts(ClTr_1(i):ClTr_2(i),2)],[clear_train_acts(ClTr_1(i):ClTr_2(i),3)],'o','Color',[rgb_colors(j)])
        
    end
end
title('Clear Training Images')
legend('Bear','Bird','Bulky Insect','Canine','Feline','Flat Fish','Flat Insect','Frog','Horse Type',...
'Octopus','Owl','Reptile','Small Fish','Small Mammal','Stick Insect','Location','best')

% Clear Testing
figure; 
hold on
for i = 1:length(ClTe_1)
    if i <= 7
        plot3([clear_test_acts(ClTe_1(i):ClTe_2(i),1)],[clear_test_acts(ClTe_1(i):ClTe_2(i),2)],[clear_test_acts(ClTe_1(i):ClTe_2(i),3)],'o','Color',def_colors(i))
    else
        j = i - length(def_colors);
        plot3([clear_test_acts(ClTe_1(i):ClTe_2(i),1)],[clear_test_acts(ClTe_1(i):ClTe_2(i),2)],[clear_test_acts(ClTe_1(i):ClTe_2(i),3)],'o','Color',[rgb_colors(j)])
        
    end
end
title('Clear Testing Images')
legend('Bear','Bird','Bulky Insect','Canine','Feline','Flat Fish','Flat Insect','Frog','Horse Type',...
'Octopus','Owl','Reptile','Small Fish','Small Mammal','Stick Insect','Location','best')
xlim([-25 25])
ylim([-15 25])

%% Mapping the Clear Training and Camo Testing Clusters

% Clear Training
figure; 
hold on
for i = 1:length(ClTr_1)
    if i <= 7
        plot3([clear_train_acts(ClTr_1(i):ClTr_2(i),1)],[clear_train_acts(ClTr_1(i):ClTr_2(i),2)],[clear_train_acts(ClTr_1(i):ClTr_2(i),3)],'o','Color',def_colors(i))
    else
        j = i - length(def_colors);
        plot3([clear_train_acts(ClTr_1(i):ClTr_2(i),1)],[clear_train_acts(ClTr_1(i):ClTr_2(i),2)],[clear_train_acts(ClTr_1(i):ClTr_2(i),3)],'o','Color',[rgb_colors(j)])
        
    end
end
title('Clear Training Images')
legend('Bear','Bird','Bulky Insect','Canine','Feline','Flat Fish','Flat Insect','Frog','Horse Type',...
'Octopus','Owl','Reptile','Small Fish','Small Mammal','Stick Insect','Location','best')

% Camo Testing
figure; 
hold on
for i = 1:length(CaTe_1)
    if i <= 7
        plot3([camo_test_acts(CaTe_1(i):CaTe_2(i),1)],[camo_test_acts(CaTe_1(i):CaTe_2(i),2)],[camo_test_acts(CaTe_1(i):CaTe_2(i),3)],'d','Color',def_colors(i))
    else
        j = i - length(def_colors);
        plot3([camo_test_acts(CaTe_1(i):CaTe_2(i),1)],[camo_test_acts(CaTe_1(i):CaTe_2(i),2)],[camo_test_acts(CaTe_1(i):CaTe_2(i),3)],'d','Color',[rgb_colors(j)])
        
    end
end
title('Camo Testing Images')
legend('Bear','Bird','Bulky Insect','Canine','Feline','Flat Fish','Flat Insect','Frog','Horse Type',...
'Octopus','Owl','Reptile','Small Fish','Small Mammal','Stick Insect','Location','best')
xlim([-25 25])
ylim([-15 25])

%% Mapping the Camo Training and Camo Testing Clusters

% Camo Training
figure; 
hold on
for i = 1:length(CaTr_1)
    if i <= 7
        plot3([camo_train_acts(CaTr_1(i):CaTr_2(i),1)],[camo_train_acts(CaTr_1(i):CaTr_2(i),2)],[camo_train_acts(CaTr_1(i):CaTr_2(i),3)],'d','Color',def_colors(i))
    else
        j = i - length(def_colors);
        plot3([camo_train_acts(CaTr_1(i):CaTr_2(i),1)],[camo_train_acts(CaTr_1(i):CaTr_2(i),2)],[camo_train_acts(CaTr_1(i):CaTr_2(i),3)],'d','Color',[rgb_colors(j)])
        
    end
end
title('Camo Training Images')
legend('Bear','Bird','Bulky Insect','Canine','Feline','Flat Fish','Flat Insect','Frog','Horse Type',...
'Octopus','Owl','Reptile','Small Fish','Small Mammal','Stick Insect','Location','best')
ylim([-15 25])

% Camo Testing
figure; 
hold on
for i = 1:length(CaTe_1)
    if i <= 7
        plot3([camo_test_acts(CaTe_1(i):CaTe_2(i),1)],[camo_test_acts(CaTe_1(i):CaTe_2(i),2)],[camo_test_acts(CaTe_1(i):CaTe_2(i),3)],'d','Color',def_colors(i))
    else
        j = i - length(def_colors);
        plot3([camo_test_acts(CaTe_1(i):CaTe_2(i),1)],[camo_test_acts(CaTe_1(i):CaTe_2(i),2)],[camo_test_acts(CaTe_1(i):CaTe_2(i),3)],'d','Color',[rgb_colors(j)])
        
    end
end
title('Camo Testing Images')
legend('Bear','Bird','Bulky Insect','Canine','Feline','Flat Fish','Flat Insect','Frog','Horse Type',...
'Octopus','Owl','Reptile','Small Fish','Small Mammal','Stick Insect','Location','best')
xlim([-20 25])
ylim([-15 25])

%% Mapping the Camo Training and Clear Testing Clusters

% Camo Training
figure; 
hold on
for i = 1:length(CaTr_1)
    if i <= 7
        plot3([camo_train_acts(CaTr_1(i):CaTr_2(i),1)],[camo_train_acts(CaTr_1(i):CaTr_2(i),2)],[camo_train_acts(CaTr_1(i):CaTr_2(i),3)],'d','Color',def_colors(i))
    else
        j = i - length(def_colors);
        plot3([camo_train_acts(CaTr_1(i):CaTr_2(i),1)],[camo_train_acts(CaTr_1(i):CaTr_2(i),2)],[camo_train_acts(CaTr_1(i):CaTr_2(i),3)],'d','Color',[rgb_colors(j)])
        
    end
end
title('Camo Training Images')
legend('Bear','Bird','Bulky Insect','Canine','Feline','Flat Fish','Flat Insect','Frog','Horse Type',...
'Octopus','Owl','Reptile','Small Fish','Small Mammal','Stick Insect','Location','best')
ylim([-15 25])

% Clear Testing
figure; 
hold on
for i = 1:length(ClTe_1)
    if i <= 7
        plot3([clear_test_acts(ClTe_1(i):ClTe_2(i),1)],[clear_test_acts(ClTe_1(i):ClTe_2(i),2)],[clear_test_acts(ClTe_1(i):ClTe_2(i),3)],'o','Color',def_colors(i))
    else
        j = i - length(def_colors);
        plot3([clear_test_acts(ClTe_1(i):ClTe_2(i),1)],[clear_test_acts(ClTe_1(i):ClTe_2(i),2)],[clear_test_acts(ClTe_1(i):ClTe_2(i),3)],'o','Color',[rgb_colors(j)])
        
    end
end
title('Clear Testing Images')
legend('Bear','Bird','Bulky Insect','Canine','Feline','Flat Fish','Flat Insect','Frog','Horse Type',...
'Octopus','Owl','Reptile','Small Fish','Small Mammal','Stick Insect','Location','best')
ylim([-15 25])


