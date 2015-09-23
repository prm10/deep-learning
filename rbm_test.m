clc;clear all;close all;
%% mnist
% load im_data.mat;
% x=batchdata;
% % x=generate_batches(data_train1,100);
% % y=generate_batches(data_validation1,100);
% % num=[1000 500 250 30];
% % [vishid,hidbiases,visbiases]=dbm_initial(x,num);
% [batchposhidprobs,vishid,hidbiases,visbiases]=rbm_model(x,1000);
% a1=batchdata(1,:,1);
% z2=a1*vishid+hidbiases;
% a2=1./(1+exp(-z2));
% z3=a2*vishid'+visbiases;
% a3=1./(1+exp(-z3));
% figure;
% imshow([reshape(a1,[28 28])' reshape(a3,[28 28])']);

%% eye(4)
% n=3;
% x=generate_batches(repmat(eye(2^n),10000,1),100);
% [batchposhidprobs,vishid,hidbiases,visbiases]=rbm_model(x,n);
% a1=eye(2^n);
% z2=a1*vishid+repmat(hidbiases,size(a1,1),1);
% a2=1./(1+exp(-z2));
% z3=a2*vishid'+repmat(visbiases,size(a1,1),1);
% a3=1./(1+exp(-z3));
% a4=(a2>.5);
% a5=sort(a4*2.^(n-1:-1:0)')'
%% eye(4)+dbm
n=3;
x=generate_batches(repmat(eye(2^n),10000,1),100);
y=eye(2^n);
num=[50 25 10 n];
% num=n;
[vishid,hidbiases,visbiases]=dbm_initial(x,num);
Weight=dbm_BP(x,y,num,vishid,hidbiases,visbiases);
% ÖØ¹¹
[a3,error]=dbm_reconstruction(eye(2^n),Weight);

% a1=eye(2^n);
% z2=a1*vishid+repmat(hidbiases,size(a1,1),1);
% a2=1./(1+exp(-z2));
% z3=a2*vishid'+repmat(visbiases,size(a1,1),1);
% a3=1./(1+exp(-z3));
% a4=(a2>.5);
% a5=sort(a4*2.^(n-1:-1:0)')'
