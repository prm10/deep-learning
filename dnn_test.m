function [data_out,err]=dnn_test(data,label,w1,b1,args)
outputway=args.outputway;
a1 = data;
[numcases,numdims]=size(data);
for i1=1:length(w1)-1
    z1=a1*w1{i1} + repmat(b1{i1},numcases,1);
    a1 = 1./(1 + exp(-z1));
end

switch(outputway)
    case 'softmax'
        M=a1*w1{end}+ repmat(b1{end},numcases,1);
        M=exp(M-max(M,[],2)*ones(1,size(M,2)));
        data_out =M./(sum(M,2)*ones(1,size(M,2)));
        err=-sum(sum(label.* log(data_out)))/numcases;
    otherwise
        data_out = 1./(1 + exp(-a1*w1{end})+ repmat(b1{end},numcases,1));
        err=-sum(sum((label-data_out).^2))/numcases/numdims; 
end
