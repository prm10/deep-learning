clc;clear all;close all;
name_str={ '富氧率','透气性指数','CO','H2','CO2','标准风速','富氧流量','冷风流量','鼓风动能','炉腹煤气量','炉腹煤气指数','理论燃烧温度','顶压','顶压2','顶压3','富氧压力','冷风压力','全压差','热风压力','实际风速','热风温度','顶温东北','顶温西南','顶温西北','顶温东南','阻力系数','鼓风湿度','设定喷煤量','本小时实际喷煤量','上小时实际喷煤量'};
load data_正常_2012-10-01.mat;
chos=[1:26];
name_str=name_str(chos);

%% 训练集、验证集、测试集
data_train0=data0(1:20000,chos);
data_test0=data0(610001:630000,chos);
%% 对输入量归一化
M_train=mean(data0(:,chos));
S_train=std(data0(:,chos));
data_train1=guiyihua(data_train0,M_train,S_train);%训练集
data_test1=guiyihua(data_test0,M_train,S_train);%测试集

% for i1=1:26
%     figure,hist(data_train1(:,i1),20);
%     title(name_str{i1});
% end
%% 剔除超限数据
range1=(max(abs(data_train1'))<5)';
range2=(max(abs(data_test1'))<5)';

data_train2=generate_img(data_train1,range1);



train_x = double(reshape(train_x',28,28,60000))/255;
test_x = double(reshape(test_x',28,28,10000))/255;
train_y = double(train_y');
test_y = double(test_y');

%% ex1 Train a 6c-2s-12c-2s Convolutional neural network 
%will run 1 epoch in about 200 second and get around 11% error. 
%With 100 epochs you'll get around 1.2% error

rand('state',0)

cnn.layers = {
    struct('type', 'i') %input layer
    struct('type', 'c', 'outputmaps', 6, 'kernelsize', 5) %convolution layer
    struct('type', 's', 'scale', 2) %sub sampling layer
    struct('type', 'c', 'outputmaps', 12, 'kernelsize', 5) %convolution layer
    struct('type', 's', 'scale', 2) %subsampling layer
};


opts.alpha = 1;
opts.batchsize = 50;
opts.numepochs = 1;

cnn = cnnsetup(cnn, train_x, train_y);
cnn = cnntrain(cnn, train_x, train_y, opts);

[er, bad] = cnntest(cnn, test_x, test_y);

%plot mean squared error
figure; plot(cnn.rL);
assert(er<0.12, 'Too big error');
