function [D]=KSVD1(X,D,iteration)
% output:D an dictionary we want to learn with size of m by p
% input :
%       X  --   denote the dataset matrix with size of m by n,
%               each column is a sample
%       D  --   the initialized Dictionary
%       lambda  --- denote a coeff of lambda in PALM

%% Fix D, solove alpha, using PALM function

lambda = 20;
for t = 1:iteration
    fprintf(['Now iteration=' num2str(t) '\n']);

    alpha = zeros(size(D,2),size(X,2));
    for i = 1:size(X,2)
        s   =  PALM(D, X(:,i), 'lambda_e', lambda);
        alpha(:,i) = single(s);
    end
    
%% Fix alpha, update D. begin
    for i=1:size(D,2)
       ai         =       alpha(i,:);
       Y          =       X-D*alpha+D(:,i)*ai;
       di         =       Y*ai';
       di         =       di./norm(di,2);
       D(:,i)     =       di;
       clear Y;
    end
end