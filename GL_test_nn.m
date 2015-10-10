clc;clear all;close all;
name_str={ '������','͸����ָ��','CO','H2','CO2','��׼����','��������','�������','�ķ綯��','¯��ú����','¯��ú��ָ��','����ȼ���¶�','��ѹ','��ѹ2','��ѹ3','����ѹ��','���ѹ��','ȫѹ��','�ȷ�ѹ��','ʵ�ʷ���','�ȷ��¶�','���¶���','��������','��������','���¶���','����ϵ��','�ķ�ʪ��','�趨��ú��','��Сʱʵ����ú��','��Сʱʵ����ú��'};
load data_¯��_2013-02-28.mat;
chos=[1:26];
name_str=name_str(chos);
%% ����ģ��
load('para10   5.mat');
data1=guiyihua(data0(:,chos),M_train,S_train);
data2=(data1-ones(size(data1,1),1)*min2)...
    ./(ones(size(data1,1),1)*(max2-min2));

[dataout2,median_out]=dnn_test(data2,w1,w2,b1,b2);
dataout1=dataout2.*(ones(size(dataout2,1),1)*(max2-min2))+ones(size(dataout2,1),1)*min2;

% for i1=1:size(dataout1,2)
%     figure;
%     plot(1:size(data1,1),data1(:,i1),1:size(data1,1),dataout1(:,i1));
%     title(name_str{i1});
% end
%% pca
% [P1,te1]=pca(data1(1:40000,:)');%ѵ��ģ��
% T1=data1*P1;
% T2=dataout1*P1;
% for i1=1:size(dataout1,2)
%     figure;
%     plot(1:size(T1,1),T1(:,i1),'b',1:size(T1,1),T2(:,i1),'r');
% end
%% median out feature analysis

% hold on;
n=1000:size(median_out,1);
for i1=1:size(median_out,2)
    figure;
    plot(n,median_out(n,i1));
end
% hold off;
% legend('1','2','3','4','5');