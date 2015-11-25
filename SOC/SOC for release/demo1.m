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
%   In set_normal_tr, set_normal_tt and set_glasses
%   each column stands for an image of 83 by 60 from AR database.
%   
%   Each entries in lab_normal_tr, lab_normal_tt and lab_glasses  
%   is the subject of corresponding image
% 
%   im_h  : the height of image
%   im_w  : the width of image
%
%   This demo is using for displays the whole algorithm and results.
%
%   For more details, please refers to 'Structured Occlusion Coding for
%   Robust Face Recognition' 
% ================================================================================



clear;clc;close all;
path = cd;
addpath([path '\database\']);
addpath([path '\utilities']);

%%%%%%%%%%%%%%%%%%%%%%%%%%%% training stage %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Loading AR database
load('set_normal_tr83_60.mat');
load('set_normal_tt83_60.mat');
load('set_glasses_83_60.mat');

%% Randomly choose occluded images of 10 subjects
ranSubjs    =    randperm(max(lab_normal_tr));
ranSubjs    =    ranSubjs(1:10);

%% Construct the original dictionary and the occluded image set
set_unOccDict   =   set_normal_tr;
lab_unOccDict   =   lab_normal_tr;

% find the indexes of selected images
indx = zeros(size(set_glasses, 2), 1);
for i = ranSubjs
    indx = indx | lab_glasses==i;
end

% Construct the occluded image set 
set_occImgs  =    set_glasses(:, indx==1);
lab_occImgs  =    lab_glasses(indx==1);

% the rest are using for testing
set_testImgs =    set_glasses(:, indx==0);
lab_testImgs =    lab_glasses(indx==0);

%% Collecting occlusion samples
[set_occSaps, lab_occSaps] = GetOccSaps(set_unOccDict, set_occImgs, 101, img_h, img_w, 0);  
% We treat the occluion sub-dictionary as 101th subject in AR database

%% Construct the occlusion dictionary

% init the parameters and occlusion dictionary
len           =    30;
iter_n        =    15;
occDict       =    randn(size(set_occSaps ,1), len);
occDict       =    occDict * diag(1./sqrt(sum(occDict.*occDict)));

% learn an occlusion dictionary from occlusion samples
set_occDict   =    KSVD1(set_occSaps, occDict, iter_n);
lab_occDict   =    lab_occSaps(1:len);

%% Compound dictionary
set_compDict  =    [set_unOccDict set_occDict];
lab_compDict  =    [lab_unOccDict; lab_occDict];
set_compDict  =    set_compDict * diag(1./sqrt(sum(set_compDict.*set_compDict)));


%%%%%%%%%%%%%%%%%%%%%%%%%%%% testing stage %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Using 'SeDuMi' solver for saving time
cvx_solver SeDuMi 

% Initiation
D       =   set_unOccDict;
B       =   set_occDict;
R       =   set_compDict;
u       =   set_testImgs(:,3);
Label   =   lab_testImgs(3);

Tn      =   zeros(1, max(lab_compDict));
for j = 1:max(lab_compDict)
    Tn(j) = length(find(lab_compDict==j));
end
%% Solve the problem (8)
w   =   StrcutSparse(R, u, Tn, 2);
x   =   w(1:size(D,2));
c   =   w(size(D,2)+1:end);

for i = 1:max(lab_unOccDict)
    r_y(i)  = norm(u - D(:,lab_unOccDict==i)*x(lab_unOccDict==i) - B*c);
end
ID = find(r_y==min(r_y));
fprintf(' the identity of testing image is %dth\n the classification result is the %dth \n', Label, ID);

% For this single occlusion scenario
r_u(1) = norm(u - D*x);

%% display
display = 1;
if display
    
    % coefficients
    figure(1);hold on;
    stem(1:size(D,2), x, '.k');
    stem(size(D,2)+1:size(D,2)+size(B,2), c, '.r');
    
    %residuals
    figure(2);hold on;
    bar(1:max(lab_unOccDict), r_y, 'k');
    bar(unique(lab_occSaps), r_u, 'r');
    
    % image
    figure(3);hold on;
    subplot(1,3,1)
    imshow(uint8(8000*reshape(u, img_h, img_w)));
    subplot(1,3,2)
    imshow(uint8(8000*reshape(D*x, img_h, img_w)));
    subplot(1,3,3)
    imshow(uint8(150+8000*reshape(B*c, img_h, img_w)));
end

