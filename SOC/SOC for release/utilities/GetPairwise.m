function pairwise = GetPairwise(m, n, lambda)
    v                  =   1:m*n;
    indxMtx            =   reshape(v', m, n);
    pairwise           =   sparse(m*n, m*n);
    v1                 =   reshape(indxMtx(1:m-1,:),(m-1)*n,1);
    v2                 =   reshape(indxMtx(2:m,:),(m-1)*n,1);
    v3                 =   reshape(indxMtx(:,1:n-1),m*(n-1),1);
    v4                 =   reshape(indxMtx(:,2:n),m*(n-1),1);

    pairwise(v1, v1+1) =   pairwise(v1, v1+1)+lambda*eye(length(v1));
    pairwise(v2, v2-1) =   pairwise(v2, v2-1)+lambda*eye(length(v2));
    pairwise(v3, v3+m) =   pairwise(v3, v3+m)+lambda*eye(length(v3));
    pairwise(v4, v4-m) =   pairwise(v4, v4-m)+lambda*eye(length(v4));
end

