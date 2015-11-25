function [ x, e ] = PALM( A, b, varargin)

% =========================================================================
% Reference:
% 'Fast l1-Minimization AlgorithmsFor Robust Face Recognition',
% Allen Y.Yang, Zihan Zhou, et al.
% =========================================================================

% params init
tol = 5e-2 ;
tol_int = 1e-6 ;
converged_ALM = 0;
maxIter_ALM = 200;
maxIter_FISTA = 400;
lambda_e = 1;
lambda_x = 1;
p = 1;
q = 1;

% Parse the optional inputs.
if (mod(length(varargin), 2) ~= 0 ),
    error(['Extra Parameters passed to the function ''' mfilename ''' parameters must be passed in pairs.']);
end
parameterCount = length(varargin)/2;

for parameterIndex = 1:parameterCount,
    parameterName = varargin{parameterIndex*2 - 1};
    parameterValue = varargin{parameterIndex*2};
    switch lower(parameterName)
        case 'tolerance'
            tol = parameterValue;
        case 'maxiteration'
            maxIter_ALM = parameterValue;
        case 'p'
            p = parameterValue;
        case 'q'
            q = parameterValue;
        case 'lambda_x'
            lambda_x = parameterValue;
        case 'lambda_e'
            lambda_e = parameterValue;
        otherwise
            error(['The parameter ''' parameterName ''' is not recognized by the function ''' mfilename '''.']);
    end
end
clear varargin


% variables init for main
[m, n] = size(A);
At = A';
G = At*A;
opts.disp = 0;
Lf = eigs(G,1,'lm',opts);
Inv_Lf = 1/Lf;
theta = zeros(m, 1);
Epsilon = 2*m/norm(b,1);
Inv_Epsilon = 1/Epsilon;

if lambda_x <10e-10
    p = 2;
end
if p==2
    const = (G+2*lambda_x*Inv_Epsilon*eye(n))\At; % for speed up
end
x = zeros(n, 1);
e = b;
k = 0;

while ~converged_ALM
    
    k = k + 1;
    
    e_pre = e;
    x_pre = x;

%   update e:  e_k+1  = min L(x_k, e, theta_k)    
    if q == 1
        e = shrink(b-A*x+Inv_Epsilon*theta, lambda_e*Inv_Epsilon);
    else if q == 2
        e = (Epsilon*(b-A*x)+theta)/(2*lambda_e+Epsilon);
        end
    end

%   update x:  x_k+1  = min L(x, e_k+1, theta_k)
    if p == 1
        % variables init for FISTA
        t = 1; z = x; w = x;
        l = 0;
        converged_FISTA = 0;
        % some constants  for speed up
        const1 = At*(b-e+Inv_Epsilon*lambda_x*theta);
        const2 = Inv_Lf*const1;
        while ~converged_FISTA
            l = l + 1;

            w_pre = w;
            w = shrink(z-Inv_Lf*(G*z)+const2, Inv_Epsilon*lambda_x*Inv_Lf);

            % two criterions to stop FISTA
            if norm(w-w_pre) < tol_int * norm(w_pre)
                converged_FISTA = 1;
            end
            if l >= maxIter_FISTA
                converged_FISTA = 1 ;
            end

            t_pre = t;
            t = 0.5*(1+sqrt(1+4*t_pre*t_pre));
            z = w + (t_pre-1)/t*(w-w_pre);
        end
        x = w;
    else if p == 2
            x = const*(b-e+Inv_Epsilon*theta);
        end
    end
%   update theta:  theta_k+1 <- theta_k    
    theta = theta + Epsilon*(b-A*x-e);
    
    % two criterions to stop PALM
    if norm([x;e] - [x_pre;e_pre]) < tol*norm([x_pre;e_pre])
        converged_ALM = 1 ;
    end
    if k >= maxIter_ALM
        converged_ALM = 1 ;
    end
    
end
end

function Y = shrink(X, alpha)
    Y = sign(X).*max(abs(X)-alpha,0);   
end

