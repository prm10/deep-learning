function [w1,w2,b1,b2,e1,e2]=dnn_train(x0,layer,active_rate,circle_times)
w1=cell(0);
w2=cell(0);
b1=cell(0);
b2=cell(0);
e1=cell(0);
i1=1;
tic;
x=x0;
for n=layer
    [x,w1{i1},w2{i1},b1{i1},b2{i1},e1{i1}]=sparse_autocoder(x,n,active_rate(i1),circle_times(i1));
    i1=i1+1;
end
[w1,w2,b1,b2,e2]=dnn_bp(x0,w1,w2,b1,b2);
toc;

