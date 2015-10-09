function data=generate_batches(x,n)
b=floor(size(x,1)/n);
data=zeros(n,size(x,2),b);
for i1=1:b
    data(:,:,i1)=x((1+(i1-1)*n):i1*n,:);
end
end