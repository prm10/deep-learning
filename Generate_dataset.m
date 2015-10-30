%% No3. 生成带标签的数据集
clc;clear;close all;
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
%% add label
% 1悬料；2滑料；3管道；4
diagnose_time={
    '2012-03-23 12:08:22'	'2012-03-23 13:05:10' 1
    '2012-03-25 12:46:13'	'2012-03-25 13:19:41' 2
    '2012-03-30 19:56:08'	'2012-03-31 00:42:21' 3
    '2012-11-16 09:39:00'	 '2012-11-16 10:29:08' 3
    '2013-01-15 23:43:11'	'2013-01-16 01:21:13' 1
    '2013-01-16 17:19:10'	'2013-01-16 19:04:44' 3
    '2013-01-25 06:18:26'	 '2013-01-25 09:54:05' 1
    '2013-02-13 14:49:30'	'2013-02-13 15:38:21' 3
    '2013-02-25 16:45:14.0'	 '2013-02-26 08:57:00' 3
    '2013-03-06 07:19:32'	'2013-03-06 15:38:04' 1
};
time_range=[datenum(diagnose_time(:,1)),datenum(diagnose_time(:,2))];
%% 
data2=[];
date2=[];
label2=[];
for i1=1:8;
    load(strcat('K:\GL_data\3\','data_',num2str(i1),'.mat'));
    outlying=max(abs((data0-ones(length(date0),1)*M)./(ones(length(date0),1)*S)),[],2)>3;
    % plot(find(outlying==0),data0(outlying==0,10),find(outlying),data0(outlying,10));
    data1=(data0-ones(length(date0),1)*M)./(ones(length(date0),1)*S)/6+0.5;
    data1=data1(~outlying,:);
    date1=date0(~outlying,:);
    label=false(length(date1),1);
    kind=zeros(length(date1),1);
    for i2=1:10
        logi=(date1>time_range(i2,1))&(date1<time_range(i2,2));
        label=label|logi;
        kind=kind+logi*diagnose_time{i2,3};
    end
%     sum(label)
%     sum(outlying)/length(outlying)
    data2=[data2;data1];
    date2=[date2;date1];
    label2=[label2;label];
    save(strcat('K:\GL_data\3\','data1_',num2str(i1),'.mat'),'data1','date1','label');
end
save('K:\GL_data\3\data2_labeled','data2','date2','label2','kind');

