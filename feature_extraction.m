function []=feature_extraction(x,L,n)
% 数据x:n*d; 最近一段时间L
% 最近一段时间内的均值、标准差
% 最近0~L、2L、4L、……时间内的原始数据、导数、二阶导数的均值、标准差
[numcases,numdims]=size(x);
if 2^n>(numcases-1)
    
end



