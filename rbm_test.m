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
x=generate_batches(repmat(eye(4),600,1),100);
[batchposhidprobs,vishid,hidbiases,visbiases]=rbm_model(x,2);
a1=eye(4);
z2=a1*vishid+repmat(hidbiases,4,1);
a2=1./(1+exp(-z2));
z3=a2*vishid'+repmat(visbiases,4,1);
a3=1./(1+exp(-z3))


