function [hout,w1,b1,b2,rec_error]=denoise_autocoder(batchdata,numhid,maxepoch)
epsilonw      = 0.01;   % Learning rate for weights 
epsilonvb     = 0.01;   % Learning rate for biases of visible units 
epsilonhb     = 0.01;   % Learning rate for biases of hidden units    
initialmomentum  = 0.5;
finalmomentum    = 0.9;

% row=0.5;%激活度
% belta=0.002;
lambda=0.0000;

[numcases, numdims, numbatches]=size(batchdata);
% Initializing symmetric weights and biases. 
w1 = 0.1*randn(numdims, numhid);
w2 = w1';%0.1*randn(numhid, numdims);
b1  = zeros(1,numhid);
b2  = zeros(1,numdims);

% Dw1  = zeros(numdims,numhid);
% Dw2  = zeros(numhid,numdims);
Dw  = zeros(numdims,numhid);
Db1 = zeros(1,numhid);
Db2 = zeros(1,numdims);
hout=zeros(numcases,numhid,numbatches);
rec_error=zeros(maxepoch,1);
std1=sqrt(sum(permute(std(batchdata),[3,2,1]).^2)/numcases);
for epoch = 1:maxepoch,
    errsum=0;
    ar=0;
    if epoch>5,
        momentum=finalmomentum;
    else
        momentum=initialmomentum;
    end;
    if mod(epoch,10)==1
        [~,median_out]=autocoder_reconstruction(batchdata(1:7,:,1),w1,b1,b2);
        plot(median_out(:,1),median_out(:,2),'*');
        drawnow;
    end
    for batch = 1:numbatches,
%% 前向传播
        data = batchdata(:,:,batch);
        % add noise
%         std1=std(a1);
        a1=data+(ones(numcases,1)*std1).*wgn(numcases,numdims,-30);%加入样本0.1倍标准差的高斯噪声
        z1=a1*w1 + repmat(b1,numcases,1);
        a2 = 1./(1 + exp(-z1));
        z2=a2*w2 + repmat(b2,numcases,1);
        a3 = 1./(1 + exp(-z2));
        
        hout(:,:,batch)=a2;
        active_rate=mean(a2);
        ar=ar+mean(active_rate);
        
        delta3=-(data-a3).*a3.*(1-a3);
        delta2=(delta3*w2').*a2.*(1-a2);
        
        db1=mean(delta2);
        db2=mean(delta3);
        dw1=data'*delta2/numcases+lambda*w1;
        dw2=a2'*delta3/numcases+lambda*w2;

%%%%%%%%% END OF NEGATIVE PHASE %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        err= sum(sum( (data-a3).^2 ));
        errsum = err + errsum;

%%%%%%%%% UPDATE WEIGHTS AND BIASES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
%         Dw1 = momentum*Dw1 - epsilonw*dw1;
%         Dw2 = momentum*Dw2 - epsilonw*dw2;
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
end;
end
