function [w1,w2,b1,b2,e]=dnn_train(x,layer,active_rate,circle_times)
w1=cell(0);
w2=cell(0);
b1=cell(0);
b2=cell(0);
e=cell(0);
i1=1;
tic;
for n=layer
    [x,w1{i1},w2{i1},b1{i1},b2{i1},e{i1}]=sparse_autocoder(x,n,active_rate(i1),circle_times);
    i1=i1+1;
end
toc;

