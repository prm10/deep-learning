clc;clear all;close all;
name_str={ '富氧率','透气性指数','CO','H2','CO2','标准风速','富氧流量','冷风流量','鼓风动能','炉腹煤气量','炉腹煤气指数','理论燃烧温度','顶压','顶压2','顶压3','富氧压力','冷风压力','全压差','热风压力','实际风速','热风温度','顶温东北','顶温西南','顶温西北','顶温东南','阻力系数','鼓风湿度','设定喷煤量','本小时实际喷煤量','上小时实际喷煤量'};
chos=[2:5,11:15,22:26];
name_str=name_str(chos);
load data_正常_2012-10-01.mat;
%% 导入数据
n=8;
l=zeros(1,n);
temp=zeros(n,30);
for i1=1:n
    load(strcat('K:\GL_data\3\','data_',num2str(i1),'.mat'));
    l(i1)=length(date0);
    temp(i1,:)=mean(data0);
end
M=l*temp/sum(l);
for i1=1:n
    load(strcat('K:\GL_data\3\','data_',num2str(i1),'.mat'));
    temp(i1,:)=mean((data0-ones(l(i1),1)*M).^2);
end
S=sqrt(l*temp/sum(l));
%% 计算变量所在区间
% i2=1;
% std2=std(data0(:,chos));
% median2=median(data0(:,chos));
% for i1=10000:10000:400000
%     d1=data0(1:i1,chos);
%     std1=std(d1);
%     median1=median(d1);
%     compare_std(i2,:)=std1./std2;
%     compare_median(i2,:)=abs(median1-median2)./std2;
%     i2=i2+1
% end
% figure;
% plot(compare_std);title('std');
% figure;
% plot(compare_median);title('median');

%% 训练集、验证集、测试集
data_train0=data0(1:20000,chos);
data_validation0=data0(1:400000,chos);
data_test0=data0(610001:630000,chos);
%% 对输入量归一化
M_train=mean(data_train0);
S_train=std(data_train0);
data_train1=guiyihua(data_train0,M_train,S_train);%训练集
data_validation1=guiyihua(data_validation0,M_train,S_train);%验证集
data_test1=guiyihua(data_test0,M_train,S_train);%测试集

% for i1=1:26
%     figure,hist(data_test1(:,i1),20);
%     title(name_str{i1});
% end

clear data0 data_train0 data_validation0 data_test0;
%% 剔除超限数据
[P1,te1]=pca(data_train1');%训练模型
T1=data_train1*P1;
T2=data_validation1*P1;
T3=data_test1*P1;
a2=T2(:,1).^2/te1(1)+T2(:,2).^2/te1(2)-10<0;
a22=(max(abs(data_validation1'))<5)';
T21=T2(a2&a22,:);
a3=T3(:,1).^2/te1(1)+T3(:,2).^2/te1(2)-10<0;
a33=(max(abs(data_test1'))<5)';
T31=T3(a3&a33,:);
% scatter(T31(:,1),T31(:,2));
data_validation2=data_validation1(a2&a22,:);
data_test2=data_test1(a3&a33,:);
min2=min([min(data_validation2);min(data_test2)]);
max2=max([max(data_validation2);max(data_test2)]);
data_validation2=(data_validation2-ones(size(data_validation2,1),1)*min2)...
    ./(ones(size(data_validation2,1),1)*(max2-min2));
data_test2=(data_test2-ones(size(data_test2,1),1)*min2)...
    ./(ones(size(data_test2,1),1)*(max2-min2));
%% 
% figure,plot(1:size(date,1),datenum(date)); %采样时间分布不均匀，周期一个小时
%% sfa
% m=5;
% x=data_train1';%训练集
% [s,W,S2]=sfa(x);%训练模型
% y=data_test1';%测试集
% [T1,Te,S1,Se]=sfa_indicater(y,W,S2,m);
% figure;
% subplot(2,2,1);
% plot(T1);title('T1');
% subplot(2,2,2);
% plot(Te);title('Te');
% subplot(2,2,3);
% plot(S1);title('S1');
% subplot(2,2,4);
% plot(Se);title('Se');
%% pca
% m=5;
% x=data_train1';%训练集
% [P,te]=pca(x);%训练模型
% y=data_test1';%测试集
% [T2,SPE]=pca_indicater(y,P,te,m);
% figure;
% subplot(2,1,1);
% plot(T2);title('T2');
% subplot(2,1,2);
% plot(SPE);title('SPE');

%% sparse autocoder
%可以分别试试用原始信号和用pca处理后信号的效果
x=generate_batches(data_validation2,100);
y=generate_batches(data_test2,100);
layer=[20 10 5];
active_rate=[0.3 0.4 0.5];
circle_times=[100 200 300];
[w1,w2,b1,b2,e]=dnn_train(x,layer,active_rate,circle_times);
save(strcat('para',num2str(layer),'.mat'),'M_train','S_train','min2','max2','w1','w2','b1','b2');
% 查看验证集和测试的重构效果
[dataout,~]=dnn_test(data_validation2,w1,w2,b1,b2);
dataout=dataout.*(ones(size(dataout,1),1)*(max2-min2))+ones(size(dataout,1),1)*min2;
T4=dataout*P1;
[dataout,~]=dnn_test(data_test2,w1,w2,b1,b2);
dataout=dataout.*(ones(size(dataout,1),1)*(max2-min2))+ones(size(dataout,1),1)*min2;
T5=dataout*P1;
% i1=13;
% figure;
% plot(1:size(data_validation2,1),data_validation2(:,i1),1:size(data_validation2,1),dataout(:,i1));
% title(name_str{i1});

for i1=1:size(dataout,2)
    figure;
    plot(1:size(T31,1),T31(:,i1),'b',1:size(T31,1),T5(:,i1),'r');
end