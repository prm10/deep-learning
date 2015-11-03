clc;close all;clear;
load data_ĞüÁÏ_2013-01-16.mat;
f=22000:size(data0,1);
data1=data0(f,:);
date1=date0(f);

plot(data1(:,:));
datestr(date1(10000),'yyyy-mm-dd HH:MM:SS')

