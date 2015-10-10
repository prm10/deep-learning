clc;clear all;close all;
name_str={ '富氧率','透气性指数','CO','H2','CO2','标准风速','富氧流量','冷风流量','鼓风动能','炉腹煤气量','炉腹煤气指数','理论燃烧温度','顶压','顶压2','顶压3','富氧压力','冷风压力','全压差','热风压力','实际风速','热风温度','顶温东北','顶温西南','顶温西北','顶温东南','阻力系数','鼓风湿度','设定喷煤量','本小时实际喷煤量','上小时实际喷煤量'};
load data_炉凉_2013-02-28.mat;
chos=[1:26];
name_str=name_str(chos);
%% 代入模型
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
% [P1,te1]=pca(data1(1:40000,:)');%训练模型
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