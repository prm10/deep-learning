function data1=guiyihua(data0,M,S)
data=data0-ones(size(data0,1),1)*M;
data1=data./(ones(size(data0,1),1)*S);
