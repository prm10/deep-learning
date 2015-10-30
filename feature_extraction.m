function []=feature_extraction(x,L,n)
% 数据x:n*d; 最近一段时间L
% 最近一段时间内的均值、标准差
% 最近0~L、2L、4L、……时间内的原始数据、导数、二阶导数的均值、标准差
[numcases]=size(x,1);
if L*2^(n-1)>(numcases-1)+2
    disp('数据长度不够');
    return;
end
xl=1+L*2^(n-1);
xmean0=zeros(n,size(x,2));
xstd0=zeros(n,size(x,2));
xmean1=zeros(n,size(x,2));
xstd1=zeros(n,size(x,2));
for i1=1:n
    xtemp=x(xl-L*2^(i1-1):xl,:);
    xmean0(i1,:)=mean(xtemp);
    xstd0(i1,:)=std(xtemp);
    
    xtemp=x(xl-L*2^(i1-1):xl,:)-x(xl-L*2^(i1-1)-1:xl-1,:);
    xmean1(i1,:)=mean(xtemp);
    xstd1(i1,:)=std(xtemp);
    
    xtemp=x(xl-L*2^(i1-1):xl,:)-2*x(xl-L*2^(i1-1)-1:xl-1,:)+x(xl-L*2^(i1-1)-2:xl-2,:);
    xmean1(i1,:)=mean(xtemp);
    xstd1(i1,:)=std(xtemp);
end


