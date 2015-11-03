function [w1,b1,e2]=dnn_bp(x,y,w1,b1,args)
% vishid=w1;
% hidvis=w2;
% hidbiases=b1;
% visbiases=b2;
%% initial
% maxepoch=10;
% batchdata=x;
% testbatchdata=y;

epsilon=0.1;
initialmomentum  = 0.5;
finalmomentum    = 0.9;

lambda=0.00001;
numdims1=size(x,2);
numdims2=size(y,2);
maxepoch=args.maxepoch;
outputway=args.outputway;
numcases=args.numcases;
e2=[];
%% data to SGD
VV=[];
Dim=[];
Weight=cell(0);
Dw=cell(0);
for i1=1:length(w1)
    weight=[w1{i1};b1{i1}];
    Weight=[Weight weight];
    VV=[VV weight(:)'];
    Dim=[Dim size(weight,1)-1];
    Dw=[Dw zeros(size(weight))];
end
Dim=[Dim Dim(1)];
lw=length(Weight);
%% start
for epoch = 1:maxepoch
    if epoch>5,
        momentum=finalmomentum;
    else
        momentum=initialmomentum;
    end;
%%%%%%%%%%%%%%%%%%%% COMPUTE TRAINING RECONSTRUCTION ERROR %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    [batchdata]=generate_batches([x,y],numcases);
    [~,~,numbatches]=size(batchdata);
    err=0;
    for batch = 1:numbatches
        data = batchdata(:,1:numdims1,batch);
        label=batchdata(:,numdims1+1:numdims1+numdims2,batch);
        [~,~,err1]=dnn_ff(data,label,Weight,outputway);
        err= err+err1;
    end
    err=err/numbatches*1000;
	e2(epoch)=err;
    fprintf(1,'BP: epoch %4i error %.4f\n',epoch,e2(epoch));
    
%     if epoch~=1
%         if e2(end-1)<err % 可以优化学习率的选取
%             epsilon=epsilon/2;
%             fprintf(1,'BP: change learning rate to %.5f \n',epsilon);
%         end
%     end
%%%%%%%%%%%%%% END OF COMPUTING TRAINING RECONSTRUCTION ERROR %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	for batch = 1:numbatches
        data = batchdata(:,1:numdims1,batch);
        label=batchdata(:,numdims1+1:numdims1+numdims2,batch);
        % 前向传播
        [a,data_out,~]=dnn_ff(data,label,Weight,outputway);
        
        % 反向传播
        delta=cell(0);
        hid=data_out;
        switch(outputway)
            case 'softmax'
                delta{length(Weight)}=-(label-hid);
            otherwise
                delta{length(Weight)}=-(label-hid).*hid.*(1-hid);
        end
        for i1=length(Weight):-1:2
            hid=a{i1}(:,1:end-1);
            weight=Weight{i1}(1:end-1,:);
            delta{i1-1}=(delta{i1}*weight').*hid.*(1-hid);
        end
        for i1=1:length(Weight)
            dw=a{i1}(:,1:end-1)'*delta{i1}/numcases+lambda*Weight{i1}(1:end-1,:);
            db=mean(delta{i1});
            Dw{i1}=momentum*Dw{i1} - epsilon*[dw;db];
            Weight{i1} = Weight{i1} + Dw{i1};
        end      
	end
end
for i1=1:length(Weight)
    weight=Weight{i1};
    w1{i1}=weight(1:end-1,:);
    b1{i1}=weight(end,:);
end
end

function [a,data_out,err]=dnn_ff(data,label,Weight,outputway)
        % 前向传播
        [numcases,numdims]=size(data);
        lw=length(Weight);
        a=cell(0);
        a{1}=[data ones(numcases,1)];
        for i1=1:lw-1
            a{i1+1} = 1./(1 + exp(-a{i1}*Weight{i1}));
            a{i1+1} = [a{i1+1} ones(numcases,1)];
        end
        switch(outputway)
            case 'softmax'
                M=a{lw}*Weight{lw};
                M=exp(M-max(M,[],2)*ones(1,size(M,2)));
                data_out =M./(sum(M,2)*ones(1,size(M,2)));
                err=-sum(sum(label.* log(data_out)))/numcases;
            otherwise
                data_out = 1./(1 + exp(-a{lw}*Weight{lw}));
                err=-sum(sum((label-data_out).^2))/numcases/numdims; 
        end
end