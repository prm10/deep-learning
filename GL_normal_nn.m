clc;clear all;close all;
name_str={ '富氧率','透气性指数','CO','H2','CO2','标准风速','富氧流量','冷风流量','鼓风动能','炉腹煤气量','炉腹煤气指数','理论燃烧温度','顶压','顶压2','顶压3','富氧压力','冷风压力','全压差','热风压力','实际风速','热风温度','顶温东北','顶温西南','顶温西北','顶温东南','阻力系数','鼓风湿度','设定喷煤量','本小时实际喷煤量','上小时实际喷煤量'};
% datestr(date(45506),'yyyy-mm-dd HH:MM:SS')

load data_正常_2012-10-01.mat;
chos=[1:26];
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
data_train0=data0(1:20000,chos);
data_validation0=data0(1:200000,chos);
data_test0=data0(500001:700000,chos);
%% 对输入量归一化
M_train=mean(data_train0);
S_train=std(data_train0);
data_train1=guiyihua(data_train0,M_train,S_train);%训练集
data_validation1=guiyihua(data_validation0,M_train,S_train);%验证集
data_test1=guiyihua(data_test0,M_train,S_train);%测试集

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
%可以分别试试用原始信号和用pca处理后信号的效果
x=generate_batches(data_validation2,100);
y=generate_batches(data_test2,100);
layer=[10 5];
active_rate=[0.4 0.5];
circle_times=[100 200];
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