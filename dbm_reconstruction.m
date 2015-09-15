function [z,z_error]=dbm_reconstruction(data,Weight)
N=size(data,1);
data = [data ones(N,1)];
wprobs=data;
for i1=1:length(Weight)
    if i1==length(Weight)/2
        wprobs = wprobs*Weight{i1}; wprobs = [wprobs  ones(N,1)];
    else
        wprobs = 1./(1 + exp(-wprobs*Weight{i1})); wprobs = [wprobs  ones(N,1)];
    end
end
z_error=sum((data(:,1:end-1)-wprobs(:,1:end-1)).^2,2)/size(data,2); 
z=wprobs(:,1:end-1);
end