function [data_out]=dnn_test(data,w1,b1)
a1 = data;
numcases=size(data,1);
for i1=1:length(w1)
    z1=a1*w1{i1} + repmat(b1{i1},numcases,1);
    a1 = 1./(1 + exp(-z1));
end
data_out=a1;
