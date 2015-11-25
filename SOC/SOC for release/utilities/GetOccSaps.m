function [set_occSaps, lab_occSaps] = GetOccSaps(set_unOccImgs, set_occImgs, lab_app, img_h, img_w, display)
    
% =========================================================================
% Reference:
% 1 - 'Face RecognitionWith Contiguous Occlusion Using Markov Random Fields',
%      Zihan Zhou, Andrew Wagner, et al.
% 2 - 'Structured Occlusion Coding for Robust Face Recognition'
%      Yandong Wen, Weiyang Liu, et al.
% =========================================================================
    
    
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
    h                =  20; % size of locality constrained dictionary
    set_occSaps      =  [];
    lab_occSaps      =  [];
    
    for i = 1:size(set_occImgs,2)
        fprintf('Obtain the %dth occlusion sample \n', i);
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
        end        
        % Extracting the occlusion pattern as occlusion sample
        e(z==1)         =  0;
        set_occSaps     =  [set_occSaps e];
        lab_occSaps     =  [lab_occSaps;lab_app];
    end
    
    %% Avoid singular and normalize
    set_occSaps   =   set_occSaps + eps;
    set_occSaps   =   set_occSaps * diag(1./sqrt(sum(set_occSaps.*set_occSaps)));
    