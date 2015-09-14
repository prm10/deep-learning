clc;clear all;close all;
name_str={ '富氧率','透气性指数','CO','H2','CO2','标准风速','富氧流量','冷风流量','鼓风动能','炉腹煤气量','炉腹煤气指数','理论燃烧温度','顶压','顶压2','顶压3','富氧压力','冷风压力','全压差','热风压力','实际风速','热风温度','顶温东北','顶温西南','顶温西北','顶温东南','阻力系数','鼓风湿度','设定喷煤量','本小时实际喷煤量','上小时实际喷煤量'};
% date_str_begin=datestr([2013,02,15,00,00,00],'yyyy-mm-dd');
% date_str_end=datestr( [2013,02,28,00,00,00],'yyyy-mm-dd');
% [date,data]=get_data_from_sql_server('[GL1].[dbo].[ZCS1]',date_str_begin,date_str_end);
% save data.mat date data;

% datestr(date(45506),'yyyy-mm-dd HH:MM:SS')

load data.mat;
% data=data(1:1000,:);
% data=data(80000:90000,:);
% for i1=1:size(data,2)
%     figure,plot(data(:,i1));title(num2str(i1));
% end
% figure,plot(data);
chos=[1,2,4,6:12,14:26];
data_train0=data(1:40000,chos);
data_test0=data(40001:end,chos);

%% 统计输入量的概率密度分布，并变换到正态分布
% figure,hist(data_train0(:,3),100);

% data_train0=data(1:20000,chos);
% data_test0=data(20001:40000,chos);
% name_str=name_str(chos);
% for i1=1:size(data_train0,2)
%     i1
%     [value,threshhold]=normalize_train(data_train0(:,i1),1000);
%     data_train1(:,i1)=normalize_test(data_train0(:,i1),value,threshhold);
%     data_test1(:,i1)=normalize_test(data_test0(:,i1),value,threshhold);
% %     figure,
% %     subplot(211);hist(data_train1(:,i1),50);title(name_str(i1));
% %     subplot(212);hist(data_test1(:,i1),50);
% end

%% 对输入量归一化
M_train=mean(data_train0);
S_train=std(data_train0);
data_train1=guiyihua(data_train0,M_train,S_train);%训练集
data_test1=guiyihua(data_test0,M_train,S_train);%测试集
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
m=5;
x=data_train1';%训练集
[P,te]=pca(x);%训练模型
y=data_test1';%测试集
[T2,SPE]=pca_indicater(y,P,te,m);
figure;
subplot(2,1,1);
plot(T2);title('T2');
subplot(2,1,2);
plot(SPE);title('SPE');