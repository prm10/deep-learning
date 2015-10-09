function [s,W,S2]=sfa(x)
 [U1,S1,V1] = svd(x*x');
 Q=S1^(-1/2)*V1';
z=Q*x;
zd=z(:,2:end)-z(:,1:end-1);
 [U2,S2,V2] = svd(zd*zd');
P=U2(:,end:-1:1);
W=P*Q;
s=W*x;
end