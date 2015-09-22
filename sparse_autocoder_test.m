clc;clear all;close all;
data_train0=eye(4);
data_train1=repmat(data_train0,1000,1);
x=generate_batches(data_train1,100);
[hout,w1,w2,b1,b2]=sparse_autocoder(x,4,500);
dataout=sparse_autocoder_reconstruction(data_train0,w1,w2,b1,b2);
i1=2;
figure;
visualize(w1)
%% dbm
% train_x=data_train1;
% rand('state',0)
% dbn.sizes = [2];
% opts.numepochs =   1;
% opts.batchsize = 100;
% opts.momentum  =   0;
% opts.alpha     =   1;
% dbn = dbnsetup(dbn, train_x, opts);
% dbn = dbntrain(dbn, train_x, opts);
% % figure; visualize(dbn.rbm{1}.W');   %  Visualize the RBM weights
% nn = dbnunfoldtonn(dbn);
% dataout=nnpredict(nn,data_train0);
% v1=data_train0;
% w=dbn.rbm{1}.W;
% b1=dbn.rbm{1, 1}.c;
% b2=dbn.rbm{1, 1}.b;
% h1 = sigm(repmat(b1', 4, 1) + v1 * w');
% v2 = sigm(repmat(b2', 4, 1) + h1 * w);
%% nn
% train_x=data_train1;
% train_y=data_train1;
% rand('state',0)
% nn = nnsetup([size(train_x,2) 2 size(train_y,2)]);
% nn.output = 'softmax';
% opts.numepochs =  1;   %  Number of full sweeps through data
% opts.batchsize = 100;  %  Take a mean gradient step over this many samples
% [nn, L] = nntrain(nn, train_x, train_y, opts);
% train_z=nnpredict(nn,data_train0);

