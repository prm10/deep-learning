function [Weight]=dbm_BP(x,y,num,vishid,hidbiases,visbiases)
%% initial
maxepoch=0;
fprintf(1,'\nFine-tuning deep autoencoder by minimizing cross entropy error. \n');
fprintf(1,'60 batches of 1000 cases each. \n');
batchdata=x;
testbatchdata=y;

[numcases numdims numbatches]=size(batchdata);
N=numcases; 
test_err=[];
train_err=[];
%% data to SGD
VV=[];
Dim=[];
Weight=cell(0);
for i1=1:length(num)
    weight=[vishid{i1};hidbiases{i1}];
    Weight=[Weight weight];
    VV=[VV weight(:)'];
    Dim=[Dim size(weight,1)-1];
end
for i1=length(num):-1:1
    weight=[vishid{i1}';visbiases{i1}];
    Weight=[Weight weight];
    VV=[VV weight(:)'];
    Dim=[Dim size(weight,1)-1];
end
Dim=[Dim Dim(1)];
%% start
for epoch = 1:maxepoch

%%%%%%%%%%%%%%%%%%%% COMPUTE TRAINING RECONSTRUCTION ERROR %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    err=0; 
    [numcases numdims numbatches]=size(batchdata);
    N=numcases;
    for batch = 1:numbatches
        data = [batchdata(:,:,batch)];
        data = [data ones(N,1)];
        wprobs=data;
        for i1=1:length(Weight)
            if i1==length(num)
                wprobs = wprobs*Weight{i1}; wprobs = [wprobs  ones(N,1)];
            else
                wprobs = 1./(1 + exp(-wprobs*Weight{i1})); wprobs = [wprobs  ones(N,1)];
            end
        end
        err= err +  1/N*sum(sum( (data(:,1:end-1)-wprobs(:,1:end-1)).^2 )); 
    end
	train_err(epoch)=err/numbatches;

%%%%%%%%%%%%%% END OF COMPUTING TRAINING RECONSTRUCTION ERROR %%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%% DISPLAY FIGURE TOP ROW REAL DATA BOTTOM ROW RECONSTRUCTIONS %%%%%%%%%%%%%%%%%%%%%%%%%
%     fprintf(1,'Displaying in figure 1: Top row - real data, Bottom row -- reconstructions \n');
%     output=[];
% 	for ii=1:15
%         output = [output data(ii,1:end-1)' dataout(ii,:)'];
%     end
%     if epoch==1 
%         close all 
%         figure('Position',[100,600,1000,200]);
%     else 
%         figure(1)
% 	end 
%     mnistdisp(output);
%     drawnow;

%%%%%%%%%%%%%%%%%%%% COMPUTE TEST RECONSTRUCTION ERROR %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    [testnumcases testnumdims testnumbatches]=size(testbatchdata);
    N=testnumcases;
    err=0;
	for batch = 1:testnumbatches
        data = testbatchdata(:,:,batch);
        data = [data ones(N,1)];
        wprobs=data;
        for i1=1:length(Weight)
            if i1==length(num)
                wprobs = wprobs*Weight{i1}; wprobs = [wprobs  ones(N,1)];
            else
                wprobs = 1./(1 + exp(-wprobs*Weight{i1})); wprobs = [wprobs  ones(N,1)];
            end
        end
        err= err +  1/N*sum(sum( (data(:,1:end-1)-wprobs(:,1:end-1)).^2 )); 
	end
    test_err(epoch)=err/testnumbatches;
    fprintf(1,'Before epoch %d Train squared error: %6.3f Test squared error: %6.3f \n',epoch,train_err(epoch),test_err(epoch));

%%%%%%%%%%%%%% END OF COMPUTING TEST RECONSTRUCTION ERROR %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

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
        max_iter=10;
        VV=[];
        for i1=1:length(Weight)
            weight=Weight{i1};
            VV=[VV weight(:)'];
        end

        [X, fX] = minimize(VV,'CG_GL',max_iter,Dim,data);
        xxx=0;
        for i1=1:length(Weight)
            Weight{i1}=reshape(X(xxx+1:xxx+(Dim(i1)+1)*Dim(i1+1)),Dim(i1)+1,Dim(i1+1));
            xxx=xxx+(Dim(i1)+1)*Dim(i1+1);
        end
%%%%%%%%%%%%%%% END OF CONJUGATE GRADIENT WITH 3 LINESEARCHES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%

	end

    save GL_weights Weight;
    save GL_error test_err train_err;

end

end