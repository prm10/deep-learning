clc;clear all;close all;
name_str={ '������','͸����ָ��','CO','H2','CO2','��׼����','��������','�������','�ķ綯��','¯��ú����','¯��ú��ָ��','����ȼ���¶�','��ѹ','��ѹ2','��ѹ3','����ѹ��','���ѹ��','ȫѹ��','�ȷ�ѹ��','ʵ�ʷ���','�ȷ��¶�','���¶���','��������','��������','���¶���','����ϵ��','�ķ�ʪ��','�趨��ú��','��Сʱʵ����ú��','��Сʱʵ����ú��'};
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

%% ͳ���������ĸ����ܶȷֲ������任����̬�ֲ�
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

%% ����������һ��
M_train=mean(data_train0);
S_train=std(data_train0);
data_train1=guiyihua(data_train0,M_train,S_train);%ѵ����
data_test1=guiyihua(data_test0,M_train,S_train);%���Լ�
%% 
% figure,plot(1:size(date,1),datenum(date)); %����ʱ��ֲ������ȣ�����һ��Сʱ
%% sfa
% m=5;
% x=data_train1';%ѵ����
% [s,W,S2]=sfa(x);%ѵ��ģ��
% y=data_test1';%���Լ�
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
x=data_train1';%ѵ����
[P,te]=pca(x);%ѵ��ģ��
y=data_test1';%���Լ�
[T2,SPE]=pca_indicater(y,P,te,m);
figure;
subplot(2,1,1);
plot(T2);title('T2');
subplot(2,1,2);
plot(SPE);title('SPE');