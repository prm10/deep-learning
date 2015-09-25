function [data_out,median_out]=dnn_test(data,w1,w2,b1,b2)
a1 = data;
numcases=size(data,1);
for i1=1:length(w1)
    z1=a1*w1{i1} + repmat(b1{i1},numcases,1);
    a1 = 1./(1 + exp(-z1));
end
median_out=a1;
a2=a1;
for i1=length(w2):-1:1
    z2=a2*w2{i1} + repmat(b2{i1},numcases,1);
    a2 = 1./(1 + exp(-z2));
end
data_out=a2;
