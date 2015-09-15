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
chos=[1:26];
% chos=[1,2,4,6:12,14:26];
% data_train0=data(1:40000,chos);
% data_test0=data(40001:end,chos);
name_str=name_str(chos);
data_train0=data(1:30000,chos);
data_validation0=data(30001:40000,chos);
data_test0=data(40001:90000,chos);
%% ����������һ��
M_train=mean(data_train0);
S_train=std(data_train0);
data_train1=guiyihua(data_train0,M_train,S_train);%ѵ����
data_validation1=guiyihua(data_validation0,M_train,S_train);%��֤��
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
% m=5;
% x=data_train1';%ѵ����
% [P,te]=pca(x);%ѵ��ģ��
% y=data_test1';%���Լ�
% [T2,SPE]=pca_indicater(y,P,te,m);
% figure;
% subplot(2,1,1);
% plot(T2);title('T2');
% subplot(2,1,2);
% plot(SPE);title('SPE');
%% dbm
m=5;
x=generate_batches(data_train1,100);
y=generate_batches(data_validation1,100);
num=[100 50 25 10];
[vishid,hidbiases,visbiases]=dbm_initial(x,num);
Weight=dbm_BP(x,y,num,vishid,hidbiases,visbiases);
% �ع�
[data_train2,data_train2_error]=dbm_reconstruction(data_train1,Weight);
[data_validation2,data_validation2_error]=dbm_reconstruction(data_validation1,Weight);
[data_test2,data_test2_error]=dbm_reconstruction(data_test1,Weight);
% figure,plot(z_error);
data_train3=iguiyihua(data_train2,M_train,S_train);%ѵ����
data_validation3=iguiyihua(data_validation2,M_train,S_train);%��֤��
data_test3=iguiyihua(data_test2,M_train,S_train);%���Լ�

data_show0=[data_train0;data_validation0];
data_show1=[data_train1;data_validation1];
data_show2=[data_train2;data_validation2];
data_show3=[data_train3;data_validation3];
for i1=1:size(data_show0,2)
    figure,plot(1:length(data_show1),data_show1(:,i1),1:length(data_show2),data_show2(:,i1));
    title(name_str{i1});
    legend('ԭʼ�ź�','�ع��ź�');
end
