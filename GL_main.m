clc;clear all;close all;
name_str={ '富氧率','透气性指数','CO','H2','CO2','标准风速','富氧流量','冷风流量','鼓风动能','炉腹煤气量','炉腹煤气指数','理论燃烧温度','顶压','顶压2','顶压3','富氧压力','冷风压力','全压差','热风压力','实际风速','热风温度','顶温东北','顶温西南','顶温西北','顶温东南','阻力系数','鼓风湿度','设定喷煤量','本小时实际喷煤量','上小时实际喷煤量'};
% date_str_begin=datestr([2013,02,15,00,00,00],'yyyy-mm-dd');
% date_str_end=datestr( [2013,02,28,00,00,00],'yyyy-mm-dd');
% [date,data]=get_data_from_sql_server('[GL1].[dbo].[ZCS1]',date_str_begin,date_str_end);
% save data.mat date data;

% datestr(date(45506),'yyyy-mm-dd HH:MM:SS')

load data_正常_2012-10-01.mat;

% data=data(1:1000,:);
% data=data(80000:90000,:);
% for i1=1:size(data,2)
%     figure,plot(data(:,i1));title(num2str(i1));
% end
% figure,plot(data);
chos=[1:26];
% chos=[1,2,4,6:12,14:26];
% data_train0=data(1:40000,chos);
% data_test0=data(40001:end,chos);
name_str=name_str(chos);
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
data_train0=data0(1:30000,chos);
data_validation0=data0(30001:40000,chos);
data_test0=data0(40001:90000,chos);
%% 对输入量归一化
M_train=mean(data_train0);
S_train=std(data_train0);
S_train=S_train*5;
data_train1=.5+guiyihua(data_train0,M_train,S_train);%训练集
data_validation1=.5+guiyihua(data_validation0,M_train,S_train);%验证集
data_test1=.5+guiyihua(data_test0,M_train,S_train);%测试集
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
%% rbm
% x=generate_batches(data_train1,100);
% y=generate_batches(data_validation1,100);
% [batchposhidprobs,vishid,hidbiases,visbiases]=rbm_model(x,1000);
% 
% a1=data_train1;
% z2=a1*vishid+repmat(hidbiases,size(a1,1),1);
% a2=1./(1+exp(-z2));
% z3=a2*vishid'+repmat(visbiases,size(a1,1),1);
% a3=1./(1+exp(-z3));
% 
% i1=3;
% plot(1:size(data_train1,1),data_train1(:,i1),1:size(data_train1,1),a3(:,i1));
%% dbm
% x=generate_batches(data_train1,100);
% y=generate_batches(data_validation1,100);
% num=[50 25 10];
% [vishid,hidbiases,visbiases]=dbm_initial(x,num);
% Weight=dbm_BP(x,y,num,vishid,hidbiases,visbiases);
% % 重构
% [data_train2,data_train2_error]=dbm_reconstruction(data_train1,Weight);
% [data_validation2,data_validation2_error]=dbm_reconstruction(data_validation1,Weight);
% [data_test2,data_test2_error]=dbm_reconstruction(data_test1,Weight);
% % figure,plot(z_error);
% data_train3=iguiyihua(data_train2,M_train,S_train);%训练集
% data_validation3=iguiyihua(data_validation2,M_train,S_train);%验证集
% data_test3=iguiyihua(data_test2,M_train,S_train);%测试集
% 
% data_show0=[data_train0;data_validation0];
% data_show1=[data_train1;data_validation1];
% data_show2=[data_train2;data_validation2];
% data_show3=[data_train3;data_validation3];
% for i1=1:size(data_show0,2)
%     figure,plot(1:length(data_show1),data_show1(:,i1),1:length(data_show2),data_show2(:,i1));
%     title(name_str{i1});
%     legend('原始信号','重构信号');
% end
%% sparse autocoder
x=generate_batches(data_train1,100);
y=generate_batches(data_validation1,100);
[hout,w1,w2,b1,b2]=sparse_autocoder(x,7,100);
dataout=sparse_autocoder_reconstruction(data_train1,w1,w2,b1,b2);
i1=2;
figure;
plot(1:size(data_train1,1),data_train1(:,i1),1:size(data_train1,1),dataout(:,i1));
