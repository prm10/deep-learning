clc;clear all;close all;
n=2;
data_train0=[%eye(2^n);%rand(7,2^n);
    .1 .2 .3 .4
    .1 .2 .3 .5
    .2 .2 .3 .4
    .3 .2 .1 .5
    ];
data_train0=eye(2^n);%rand(7,2^n);
label0=[1;1;0;0];
% data_train0=[data_train0;[1 0 0 .1]];
% data_train0(1,1:2)=[1,0];
% data_train0(2,1:2)=[1,0.8];
data_train1=repmat(data_train0,10000,1);
label1=repmat(label0,10000,1);
x=generate_batches(data_train1,100);
y=generate_batches(label1,100);
%% sparse
% [hout,w1,b1,b2,rec_error]=sparse_autocoder(x,n,200);
% [hout,w1,b1,b2,rec_error]=contractive_autocoder(x,n,200);
% [hout,w1,b1,b2,rec_error]=denoise_autocoder(x,n,200);
% [dataout,median_out]=autocoder_reconstruction(data_train0,w1,b1,b2);

%% dnn
layer=[2*n n];
circle_times=[200 200];
[w1,b1,b2]=dnn_train(x,y,layer,circle_times);
[dataout]=dnn_test(data_train0,w1,b1);
%% result visualize
% figure;
% imshow([data_train0 dataout]);
% info=length(unique((median_out>.5)*2.^(n-1:-1:0)'));

% figure;
% visualize(w1{1})
