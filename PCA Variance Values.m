%% PCA Variance Values
% Finding the PCA variance values for 3 different network PCAs to
% demonstrate why the first 3 dimensions are used for feature spaces

%%

load('exp_camo_net.mat')
load('exp_clear_net.mat')
load('camo_net.mat')
load('clear_net.mat')

load('CamoTestds2.mat')

layer = 'fc';
ExpCamoNetCamofeaturesTest = activations(exp_camo_net,CamoTestds2,layer,'OutputAs','rows');
ExpClearNetCamofeaturesTest = activations(exp_clear_net,CamoTestds2,layer,'OutputAs','rows');
CamoNetCamofeaturesTest = activations(camo_net,CamoTestds2,layer,'OutputAs','rows');
ClearNetCamofeaturesTest = activations(clear_net,CamoTestds2,layer,'OutputAs','rows');

[coeff1,score1,latent1] = pca(ExpCamoNetCamofeaturesTest);
[coeff2,score2,latent2] = pca(ExpClearNetCamofeaturesTest);
[coeff3,score3,latent3] = pca(CamoNetCamofeaturesTest);
[coeff4,score4,latent4] = pca(ClearNetCamofeaturesTest);

diff(latent1)
diff(latent2)
diff(latent3)
diff(latent4)















