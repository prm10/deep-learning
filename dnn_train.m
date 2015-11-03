function [w1,b1,b2,w11,b11,e1,e2]=dnn_train(x0,y0,args)
w1=cell(0);
b1=cell(0);
e1=cell(0);
i1=1;
tic;
x=x0;
layer=args.layer;
maxepoch=args.maxepoch;
outputway=args.outputway;
numcases=args.numcases;
for n=layer
    [x,w1{i1},b1{i1},b2{i1},e1{i1}]=contractive_autocoder(x,n,args);
    i1=i1+1;
end
[numcases, numdims, numbatches]=size(y0);
w11=[w1 0.1*randn(size(w1{end},2),numdims);];
b11=[b1 zeros(1,numdims)];
[w11,b11,e2]=dnn_bp(x0,y0,w11,b11,args);
toc;

