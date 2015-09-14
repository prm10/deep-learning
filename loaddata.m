function [u,z]=loaddata(str)
data=importdata(strcat('TE_process/',str,'.dat'));
if size(data,1)~=52
    data=data';
end
u=data(1:41,:);
z=data(42:52,:);
end