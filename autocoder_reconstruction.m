function [a3,a2]=autocoder_reconstruction(data_train1,w1,b1,b2)
a1 = data_train1;
numcases=size(data_train1,1);
z1=a1*w1 + repmat(b1,numcases,1);
a2 = 1./(1 + exp(-z1));
z2=a2*w1' + repmat(b2,numcases,1);
a3 = 1./(1 + exp(-z2));

