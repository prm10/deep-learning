function data=generate_batches(x,n)
x=x(randperm(size(x,1)),:);
b=ceil(size(x,1)/n);
data=zeros(n,size(x,2),b);
if b*n>size(x,1)
    x=[x;x(randperm(size(x,1),b*n-size(x,1)),:)];
end
for i1=1:b
    data(:,:,i1)=x((1+(i1-1)*n):i1*n,:);
end
end