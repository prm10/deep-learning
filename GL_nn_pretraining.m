clc;clear;close all;
name_str={ '富氧率','透气性指数','CO','H2','CO2','标准风速','富氧流量','冷风流量','鼓风动能','炉腹煤气量','炉腹煤气指数','理论燃烧温度','顶压','顶压2','顶压3','富氧压力','冷风压力','全压差','热风压力','实际风速','热风温度','顶温东北','顶温西南','顶温西北','顶温东南','阻力系数','鼓风湿度','设定喷煤量','本小时实际喷煤量','上小时实际喷煤量'};
% chos=[2:5,11:15,22:26];
% name_str=name_str(chos);
% load data_正常_2012-10-01.mat;
%% 导入数据
load('K:\GL_data\3\data2_labeled');
%% autocoder
%可以分别试试用原始信号和用pca处理后信号的效果
x=generate_batches(data2,100);
y=generate_batches(label2,100);
clear data2 date2 label2 kind
layer=[100 50 20];
circle_times=[200 200 300];
[w1,b1,b2,w11,b11]=dnn_train(x,y,layer,circle_times);
save(strcat('para',num2str(layer),'.mat'),'w1','b1','b2','w11','b11');
% 查看验证集和测试的重构效果
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