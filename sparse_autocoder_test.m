clc;clear all;close all;
n=4;
data_train0=eye(2^n);%rand(2^n);
data_train1=repmat(data_train0,10000,1);
x=generate_batches(data_train1,100);
%% sparse
% [~,w1,w2,b1,b2]=sparse_autocoder(x,n,200);
% [dataout,median_out]=sparse_autocoder_reconstruction(data_train0,w1,w2,b1,b2);

%% dnn
layer=[n];
active_rate=[0.5];
circle_times=[10];
[w1,w2,b1,b2]=dnn_train(x,layer,active_rate,circle_times);
[dataout,median_out]=dnn_test(data_train0,w1,w2,b1,b2);
%% result visualize
imshow([data_train0 dataout]);
info=length(unique((median_out>.5)*2.^(n-1:-1:0)'));
% figure;
% visualize(w1)
