function [vishid,hidbiases,visbiases]=dbm_initial(batchdata,num)
dataIn=batchdata;
vishid=cell(0);
hidbiases=cell(0);
visbiases=cell(0);
for i1=1:length(num)
    [dataIn,vishid{i1},hidbiases{i1},visbiases{i1}]=rbm_model(dataIn,num(i1));
end
end