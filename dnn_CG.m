function [f, df] = dnn_CG(VV,Dim,XX)
N = size(XX,1);
Weight=cell(0);
xxx=0;
for i1=1:length(Dim)-1
    Weight{i1}=reshape(VV(xxx+1:xxx+(Dim(i1)+1)*Dim(i1+1)),Dim(i1)+1,Dim(i1+1));
    xxx=xxx+(Dim(i1)+1)*Dim(i1+1);
end
 
XX = [XX ones(N,1)];
wprobs=cell(0);
wprobs{1}=XX;
for i1=1:length(Weight)
    wprobs{i1+1} = 1./(1 + exp(-wprobs{i1}*Weight{i1}));
    wprobs{i1+1} = [wprobs{i1+1}  ones(N,1)];
end
Xout=wprobs{length(Dim)};
XXout=Xout(:,1:end-1);

f = -1/N*sum(sum( XX(:,1:end-1).*log(XXout) + (1-XX(:,1:end-1)).*log(1-XXout)));

Ix=cell(0);
dw=cell(0);

IO = 1/N*(XXout-XX(:,1:end-1));
Ix{length(Weight)}=IO;
dw{length(Weight)} =  wprobs{length(Weight)}'*Ix{length(Weight)};
for i1=length(Weight)-1:1
    bprobs = (Ix{i1+1}*Weight{i1+1}').*wprobs{i1+1}.*(1-wprobs{i1+1});
    Ix{i1}=bprobs(:,1:end-1);
    dw{i1} =  wprobs{i1}'*Ix{i1};
end
df=[];
for i1=1:length(Weight)
    dw1=dw{i1};
	df=[df;dw1(:)];
end