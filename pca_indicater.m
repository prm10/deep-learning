function [T2,SPE,t]=pca_indicater(y,P,te,m)
t=P(:,1:m)'*y;
T2=sum((te(1:m,1:m)^-1*t).*t);
SPE=sum(((eye(size(y,1))-P(:,1:m)*P(:,1:m)')*y).^2);
end