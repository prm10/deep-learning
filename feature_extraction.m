function []=feature_extraction(x,L,n)
% ����x:n*d; ���һ��ʱ��L
% ���һ��ʱ���ڵľ�ֵ����׼��
% ���0~L��2L��4L������ʱ���ڵ�ԭʼ���ݡ����������׵����ľ�ֵ����׼��
[numcases]=size(x,1);
if L*2^(n-1)>(numcases-1)+2
    disp('���ݳ��Ȳ���');
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


