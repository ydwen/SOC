% ===============================================================================
%   Reference:
%   
%   Structured Occlusion Coding for Robust Face Recognition,
%   Yandong Wen, Weiyang Liu, Meng Yang, Yuli Fu, Youjun Xiang and Rui Hu
%  
%   Written by Yandong Wen @ SCUT
%   Feb, 2015
% ===============================================================================

% ===============================================================================
%   Explaination:
%
%   There are 1399 normal images in AR database, with 14 images in a subject. 
%   We randomly choose 7 images from each subject to construct the original 
%   dictionary. 
%   
%   In set_normal_tr and set_glasses
%   each column stands for an image of 83 by 60 from AR database.
%   
%   Each entries in lab_normal_tr, lab_normal_tt and lab_glasses  
%   is the subject of corresponding image
% 
%   im_h  : the height of image
%   im_w  : the width of image
%
%   This demo is using for showing how we estimate the occlusion pattern.
%
%   For more details, please refers to 'Structured Occlusion Coding for
%   Robust Face Recognition'
% ================================================================================



clear;clc;close all;
path = cd;
addpath([path '\database\']);
addpath([path '\utilities']);

% Loading AR database
load('set_normal_tr83_60.mat');
load('set_glasses_83_60.mat');


%% Construct the original dictionary 
set_unOccImgs       =    set_normal_tr;
set_occImgs         =    set_glasses;

% normalize
set_unOccImgs       =    set_unOccImgs * diag(1./sqrt(sum(set_unOccImgs.*set_unOccImgs)));
set_occImgs         =    set_occImgs * diag(1./sqrt(sum(set_occImgs.*set_occImgs)));

%% Graph Cut Parameters
beta = 20;
GC_param.tao        =    0.005:-0.0005:0.002;
GC_param.maxIter    =    length(GC_param.tao);

GC_param.pairwise   =    GetPairwise(img_h, img_w, beta);
GC_param.unary      =    zeros(2, img_h*img_w);
GC_param.labelcost  =    [0 1; 1 0];

%% Get occlusion samples
h         =    20; % size of locality constrained dictionary
i         =    3;
display   =    1;

% z==1 means unocclusion pixel. z==0 means occlusion pixel.
z         =    ones(size(set_unOccImgs, 1), 1);
D         =    set_unOccImgs;
u         =    set_occImgs(:,i);
psi       =    abs(D'*u);
psi_sort  =    sort(psi,'descend');
D         =    D(:,psi>=psi_sort(h)); % D turns into locality constrained dictionary


for j = 1:GC_param.maxIter
   %% exluding the occluded pixels
    sup      =    z;
    D_star   =    D(sup==1,:);
    y_star   =    u(sup==1,:);

   %% err update, using PALM to minimize ||e_star||_1, s.t. u_star = D_star*x + e_star
    [x ~]    =    PALM(D_star, y_star, 'p', 1, 'q', 1, 'lambda_x', 0);
    e        =    u - D*x;
    
   %% sup update, using graph cuts
    [GC_param.unary(1,:), GC_param.unary(2,:)]  =  ...
                    logLikeli(abs(e), GC_param.tao(j));
    [z, ~, ~]  =  GCMex(sup, single(GC_param.unary), ...
                    GC_param.pairwise, single(GC_param.labelcost), 0);
                
    %% display
    if display
        figure(1);
        % occlusion mask
        subplot(2, GC_param.maxIter+1, j+1);
        imshow(uint8(255*reshape(z, img_h, img_w)));
        
        % occlusion sample
        e(z==1)  =  0;
        subplot(2, GC_param.maxIter+1, GC_param.maxIter+1+j+1);
        imshow(uint8(150+8000*reshape(e, img_h, img_w)));
        
        % occluded image
        subplot(2, GC_param.maxIter+1, GC_param.maxIter+1+1);
        imshow(uint8(8000*reshape(u, img_h, img_w)));
    end
    
end
