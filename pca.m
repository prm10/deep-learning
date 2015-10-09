function [P,te]=pca(x)
[U1,S1,~] = svd(x*x'/(size(x,2)-1));
te=diag(S1);
P=U1;
end