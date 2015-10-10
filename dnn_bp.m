function [w1,w2,b1,b2,e2]=dnn_bp(x,w1,w2,b1,b2)
% vishid=w1;
% hidvis=w2;
% hidbiases=b1;
% visbiases=b2;
%% initial
maxepoch=10;
batchdata=x;
% testbatchdata=y;

[numcases numdims numbatches]=size(batchdata);
% test_err=[];
e2=[];
%% data to SGD
VV=[];
Dim=[];
Weight=cell(0);
for i1=1:length(w1)
    weight=[w1{i1};b1{i1}];
    Weight=[Weight weight];
    VV=[VV weight(:)'];
    Dim=[Dim size(weight,1)-1];
end
for i1=length(w1):-1:1
    weight=[w2{i1};b2{i1}];
    Weight=[Weight weight];
    VV=[VV weight(:)'];
    Dim=[Dim size(weight,1)-1];
end
Dim=[Dim Dim(1)];
%% start
for epoch = 1:maxepoch

%%%%%%%%%%%%%%%%%%%% COMPUTE TRAINING RECONSTRUCTION ERROR %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    err=0;
    for batch = 1:numbatches
        data = batchdata(:,:,batch);
        wprobs=[data ones(numcases,1)];
        for i1=1:length(Weight)
            wprobs = 1./(1 + exp(-wprobs*Weight{i1}));
            wprobs = [wprobs ones(numcases,1)];
        end
        data_out=wprobs(:,1:end-1);
        err= err+sum(sum((data-data_out).^2))/numcases/numdims; 
    end
	e2(epoch)=err/numbatches*1000;
    fprintf(1,'BP: epoch %d batch %.4f\n',epoch,e2(epoch));
%%%%%%%%%%%%%% END OF COMPUTING TRAINING RECONSTRUCTION ERROR %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    tt=0;
	for batch = 1:numbatches/10
%         fprintf(1,'epoch %d batch %d\r',epoch,batch);

%%%%%%%%%%% COMBINE 10 MINIBATCHES INTO 1 LARGER MINIBATCH %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        tt=tt+1; 
        data=[];
        for kk=1:10
            data=[data 
            batchdata(:,:,(tt-1)*10+kk)]; 
        end 

%%%%%%%%%%%%%%% PERFORM CONJUGATE GRADIENT WITH 3 LINESEARCHES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        a=cell(0);
        a{1}=[data ones(numcases*10,1)];
        for i1=1:length(Weight)
            a{i1+1} = 1./(1 + exp(-a{i1}*Weight{i1}));
            a{i1+1} = [a{i1+1} ones(numcases*10,1)];
        end
        Xout=a{length(Weight)+1};
        data_out=Xout(:,1:end-1);
        err= err+sum(sum((data-data_out).^2))/numcases/numdims; 
        
       %% ·´Ïò´«²¥
        delta=cell(0);
        hid=a{length(Weight)+1};
        hid=hid(:,1:end-1);
        delta{length(Weight)}=-(data-hid).*hid.*(1-hid);
        for i1=length(Weight):1
            hid=a{i1};
            hid=hid(:,1:end-1);
            delta{i1}=-(data-hid).*hid.*(1-hid);
        end
       
%%%%%%%%%%%%%%% END OF CONJUGATE GRADIENT WITH 3 LINESEARCHES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%

	end

%     save dnn_weights Weight;
%     save dnn_error train_err;

end
for i1=1:length(Weight)/2
    weight=Weight{i1};
    w1{i1}=weight(1:end-1,:);
    b1{i1}=weight(end,:);
end
for i1=1:length(Weight)/2
    weight=Weight{length(Weight)+1-i1};
    w2{i1}=weight(1:end-1,:);
    b2{i1}=weight(end,:);
end
end