function [w1,w2,b1,b2,e2]=dnn_bp(x,w1,w2,b1,b2,maxepoch)
% vishid=w1;
% hidvis=w2;
% hidbiases=b1;
% visbiases=b2;
%% initial
% maxepoch=10;
batchdata=x;
% testbatchdata=y;

epsilon=0.01;
initialmomentum  = 0.5;
finalmomentum    = 0.9;
% row=0.5;%激活度
belta=0.002;
lambda=0.00005;

[numcases numdims numbatches]=size(batchdata);
% test_err=[];
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
for i1=length(w1):-1:1
    weight=[w2{i1};b2{i1}];
    Weight=[Weight weight];
    VV=[VV weight(:)'];
    Dim=[Dim size(weight,1)-1];
    Dw=[Dw zeros(size(weight))];
end
Dim=[Dim Dim(1)];
%% start
for epoch = 1:maxepoch
    if epoch>5,
        momentum=finalmomentum;
    else
        momentum=initialmomentum;
    end;
%%%%%%%%%%%%%%%%%%%% COMPUTE TRAINING RECONSTRUCTION ERROR %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    err=0;
    for batch = 1:numbatches
        data = batchdata(:,:,batch);
        wprobs=[data ones(numcases,1)];
        for i1=1:length(Weight)
            wprobs = [1./(1 + exp(-wprobs*Weight{i1})) ones(numcases,1)];
        end
        data_out=wprobs(:,1:end-1);
        err= err+sum(sum((data-data_out).^2))/numcases/numdims;
    end
	e2(epoch)=err/numbatches*1000;
    fprintf(1,'BP: epoch %4i error %.4f\n',epoch,e2(epoch));
    
    if epoch~=1
        if err0<err
            epsilon=epsilon/2;
            fprintf(1,'BP: change learning rate to %.5f',epsilon);
        end
    end
    err0=err;
%%%%%%%%%%%%%% END OF COMPUTING TRAINING RECONSTRUCTION ERROR %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    tt=0;
	for batch = 1:numbatches/10
%%%%%%%%%%% COMBINE 10 MINIBATCHES INTO 1 LARGER MINIBATCH %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        tt=tt+1; 
        data=[];
        for kk=1:10
            data=[data;batchdata(:,:,(tt-1)*10+kk)]; 
        end
        numcases2=numcases*10;

%%%%%%%%%%%%%%% PERFORM CONJUGATE GRADIENT WITH 3 LINESEARCHES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        a=cell(0);
        a{1}=[data ones(numcases2,1)];
        for i1=1:length(Weight)
            a{i1+1} = 1./(1 + exp(-a{i1}*Weight{i1}));
            a{i1+1} = [a{i1+1} ones(numcases2,1)];
        end
        Xout=a{length(Weight)+1};
        data_out=Xout(:,1:end-1);
        err= err+sum(sum((data-data_out).^2))/numcases2/numdims; 
        
       %% 反向传播
        delta=cell(0);
        hid=a{length(Weight)+1}(:,1:end-1);
        delta{length(Weight)}=-(data-hid).*hid.*(1-hid);
        for i1=length(Weight):-1:2
            hid=a{i1}(:,1:end-1);
            weight=Weight{i1}(1:end-1,:);
            delta{i1-1}=(delta{i1}*weight').*hid.*(1-hid);
        end
        for i1=1:length(Weight)
            dw=a{i1}(:,1:end-1)'*delta{i1}/numcases2+lambda*Weight{i1}(1:end-1,:);
            db=mean(delta{i1});
            Dw{i1}=momentum*Dw{i1} - epsilon*[dw;db];
            Weight{i1} = Weight{i1} + Dw{i1};
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