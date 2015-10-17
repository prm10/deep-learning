clc;clear all;close all;
name_str={ '������','͸����ָ��','CO','H2','CO2','��׼����','��������','�������','�ķ綯��','¯��ú����','¯��ú��ָ��','����ȼ���¶�','��ѹ','��ѹ2','��ѹ3','����ѹ��','���ѹ��','ȫѹ��','�ȷ�ѹ��','ʵ�ʷ���','�ȷ��¶�','���¶���','��������','��������','���¶���','����ϵ��','�ķ�ʪ��','�趨��ú��','��Сʱʵ����ú��','��Сʱʵ����ú��'};
load data_����_2012-10-01.mat;
chos=[1:26];
name_str=name_str(chos);

%% ѵ��������֤�������Լ�
data_train0=data0(1:20000,chos);
data_test0=data0(610001:630000,chos);
%% ����������һ��
M_train=mean(data0(:,chos));
S_train=std(data0(:,chos));
data_train1=guiyihua(data_train0,M_train,S_train);%ѵ����
data_test1=guiyihua(data_test0,M_train,S_train);%���Լ�

% for i1=1:26
%     figure,hist(data_train1(:,i1),20);
%     title(name_str{i1});
% end
%% �޳���������
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
