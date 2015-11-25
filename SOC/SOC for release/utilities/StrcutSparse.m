function [x] = StrcutSparse(D, y, Tn, q)

beta = sqrt(Tn);

n = length(Tn);
N = sum(Tn);

cvx_begin quiet
variable x(N,1)
t1 = 0;
ind = 0;
for i = 1:n
    t1 = t1 + norm( x(ind+1:ind+Tn(i))*beta(i) , q );
    ind = ind + Tn(i);
end
minimize(0.01*t1 + norm(y - D * x))

cvx_end
