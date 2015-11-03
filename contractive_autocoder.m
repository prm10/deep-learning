function [hout,w1,b1,b2,rec_error]=contractive_autocoder(x,numhid,args)
epsilonw      = 0.01;   % Learning rate for weights 
epsilonvb     = 0.01;   % Learning rate for biases of visible units 
epsilonhb     = 0.01;   % Learning rate for biases of hidden units    
initialmomentum  = 0.5;
finalmomentum    = 0.9;
lambda=0.00001;
maxepoch=args.maxepoch;
numcases=args.numcases;
numdims=size(x,2);
[batchdata]=generate_batches(x,numcases);
[~, ~, numbatches]=size(batchdata);
% Initializing symmetric weights and biases. 
w1 = 0.1*randn(numdims, numhid);
w2 = w1';%0.1*randn(numhid, numdims);
b1  = zeros(1,numhid);
b2  = zeros(1,numdims);

Dw  = zeros(numdims,numhid);
Db1 = zeros(1,numhid);
Db2 = zeros(1,numdims);
rec_error=zeros(maxepoch,1);
for epoch = 1:maxepoch,
    errsum=0;
    ar=0;
    if epoch>5,
        momentum=finalmomentum;
    else
        momentum=initialmomentum;
    end;
    [batchdata]=generate_batches(x,numcases);
    [~, ~, numbatches]=size(batchdata);
    for batch = 1:numbatches,
%% Ç°Ïò´«²¥
        a1 = batchdata(:,:,batch);
        z1=a1*w1 + repmat(b1,numcases,1);
        a2 = 1./(1 + exp(-z1));
        z2=a2*w2 + repmat(b2,numcases,1);
        a3 = 1./(1 + exp(-z2));
        
        active_rate=mean(a2);
        ar=ar+mean(active_rate);
        
        delta3=-(a1-a3).*a3.*(1-a3);
        delta2=(delta3*w2').*a2.*(1-a2);
        
        g0=(a2.*(1-a2)).^2;
        g1=mean((1-2*a2).*g0).*sum(w1.^2);
        g2=(ones(numdims,numcases)*g0)/numcases.*w1+(a1'*((1-2*a2).*g0))/numcases.*(ones(numdims,1)*sum(w1.^2));
        
        db1=mean(delta2)+lambda*g1;
        db2=mean(delta3);
        dw1=a1'*delta2/numcases+lambda*g2;
        dw2=a2'*delta3/numcases+lambda*w2;

%%%%%%%%% END OF NEGATIVE PHASE %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        err= sum(sum( (a1-a3).^2 ));
        errsum = err + errsum;

%%%%%%%%% UPDATE WEIGHTS AND BIASES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
        Dw=momentum*Dw - epsilonw*(dw1+dw2')/2;
        Db1 = momentum*Db1 - epsilonhb*db1;
        Db2 = momentum*Db2 - epsilonvb*db2;

        w1 = w1 + Dw;
        w2 = w2 + Dw';
        b1 = b1 + Db1;
        b2 = b2 + Db2;
%%%%%%%%%%%%%%%% END OF UPDATES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 

    end
    errsum=errsum/numcases/numdims/numbatches;
    rec_error(epoch)=errsum*1e3;
    fprintf(1, 'epoch %4i error %.4f active rate %3.3f \n', epoch, rec_error(epoch), ar/numbatches); 
end
z1=x*w1 + repmat(b1,size(x,1),1);
hout = 1./(1 + exp(-z1));
end
