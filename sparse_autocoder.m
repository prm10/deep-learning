function [hout,w1,w2,b1,b2]=sparse_autocoder(batchdata,numhid,maxepoch)
epsilonw      = 0.01;   % Learning rate for weights 
epsilonvb     = 0.01;   % Learning rate for biases of visible units 
epsilonhb     = 0.01;   % Learning rate for biases of hidden units    
initialmomentum  = 0.5;
finalmomentum    = 0.9;

row=0.5;%�����
belta=0.01;

[numcases, numdims, numbatches]=size(batchdata);
% Initializing symmetric weights and biases. 
w1 = 0.1*randn(numdims, numhid);
w2 = 0.1*randn(numhid, numdims);
b1  = zeros(1,numhid);
b2  = zeros(1,numdims);

Dw1  = zeros(numdims,numhid);
Dw2  = zeros(numhid,numdims);
Db1 = zeros(1,numhid);
Db2 = zeros(1,numdims);
hout=zeros(numcases,numhid,numbatches);
for epoch = 1:maxepoch,
    errsum=0;
    ar=0;
    for batch = 1:numbatches,
%%%%%%%%% START POSITIVE PHASE %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        a1 = batchdata(:,:,batch);
        z1=a1*w1 + repmat(b1,numcases,1);
        a2 = 1./(1 + exp(-z1));
        z2=a2*w2 + repmat(b2,numcases,1);
        a3 = 1./(1 + exp(-z2));
        
        hout(:,:,batch)=a2;
        active_rate=mean(a2);
        ar=ar+mean(active_rate);
        
        delta3=-(a1-a3).*a3.*(1-a3);
        delta2=(delta3*w2'+belta*ones(numcases,1)*(-row./active_rate+(1-row)./(1-active_rate))).*a2.*(1-a2);
        
        db1=mean(delta2);
        db2=mean(delta3);
        dw1=a1'*delta2/numcases;
        dw2=a2'*delta3/numcases;

%%%%%%%%% END OF NEGATIVE PHASE %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        err= sum(sum( (a1-a3).^2 ));
        errsum = err + errsum;

        if epoch>5,
            momentum=finalmomentum;
        else
            momentum=initialmomentum;
        end;

%%%%%%%%% UPDATE WEIGHTS AND BIASES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
        Dw1 = momentum*Dw1 - epsilonw*dw1;
        Dw2 = momentum*Dw2 - epsilonw*dw2;
        Db1 = momentum*Db1 - epsilonhb*db1;
        Db2 = momentum*Db2 - epsilonvb*db2;

        w1 = w1 + Dw1;
        w1 = w1 + Dw1;
        b1 = b1 + Db1;
        b2 = b2 + Db2;
%%%%%%%%%%%%%%%% END OF UPDATES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 

    end
    fprintf(1, 'epoch %4i error %6.1f active rate %3.3f \n', epoch, errsum, ar/numbatches); 
end;
end