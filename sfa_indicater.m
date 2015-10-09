function [T1,Te,S1,Se]=sfa_indicater(x,W,S2,m)
s=W*x;
sd=s(:,2:end)-s(:,1:end-1);
T1=sum(s(1:m,:).*s(1:m,:));
Te=sum(s(m+1:end,:).*s(m+1:end,:));
S1=sum((S2(1:m,1:m)^-1*sd(1:m,:)).*sd(1:m,:));
Se=sum((S2(m+1:end,m+1:end)^-1*sd(m+1:end,:)).*sd(m+1:end,:));
end