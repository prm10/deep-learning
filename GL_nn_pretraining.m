clc;clear;close all;
name_str={ '������','͸����ָ��','CO','H2','CO2','��׼����','��������','�������','�ķ綯��','¯��ú����','¯��ú��ָ��','����ȼ���¶�','��ѹ','��ѹ2','��ѹ3','����ѹ��','���ѹ��','ȫѹ��','�ȷ�ѹ��','ʵ�ʷ���','�ȷ��¶�','���¶���','��������','��������','���¶���','����ϵ��','�ķ�ʪ��','�趨��ú��','��Сʱʵ����ú��','��Сʱʵ����ú��'};
% chos=[2:5,11:15,22:26];
% name_str=name_str(chos);
% load data_����_2012-10-01.mat;
%% ��������
load('K:\GL_data\3\data2_labeled');
%% autocoder
%���Էֱ�������ԭʼ�źź���pca������źŵ�Ч��
x=generate_batches(data2,100);
y=generate_batches(label2,100);
clear data2 date2 label2 kind
layer=[100 50 20];
circle_times=[200 200 300];
[w1,b1,b2,w11,b11]=dnn_train(x,y,layer,circle_times);
save(strcat('para',num2str(layer),'.mat'),'w1','b1','b2','w11','b11');
% �鿴��֤���Ͳ��Ե��ع�Ч��
clear x y;
load('K:\GL_data\3\data2_labeled');
[dataout]=dnn_test(data2,w11,b11);
n=1:length(dataout);
figure;
plot(n,label2,n,dataout);
W=w1;
for i1=length(w1):-1:1
    W=[W w1{i1}'];
end
B=b1;
for i1=length(b2):-1:1
    B=[B b2{i1}];
end
[dataout]=dnn_test(data2,W,B);
error=mean((data2-dataout).^2,2);
figure;
plot(n,label2,n,error);
% dataout=dataout.*(ones(size(dataout,1),1)*(max2-min2))+ones(size(dataout,1),1)*min2;
% T4=dataout*P1;
% [dataout,~]=dnn_test(data_test2,w1,w2,b1,b2);
% dataout=dataout.*(ones(size(dataout,1),1)*(max2-min2))+ones(size(dataout,1),1)*min2;
% T5=dataout*P1;
% % i1=13;
% % figure;
% % plot(1:size(data_validation2,1),data_validation2(:,i1),1:size(data_validation2,1),dataout(:,i1));
% % title(name_str{i1});
% 
% for i1=1:size(dataout,2)
%     figure;
%     plot(1:size(T31,1),T31(:,i1),'b',1:size(T31,1),T5(:,i1),'r');
% end