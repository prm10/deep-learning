function data2=generate_img(data1,range1,leng,stride)
a1=find(range1==0);
a1=[0;a1;size(data1,1)+1];
i0=1;
for i1=2:length(a1);
    if a1(i1)-a1(i1-1)-1>leng
        for i2=1:floor((a1(i1)-a1(i1-1)-leng-1)/stride+1)
            data2(:,:,i0)=data1(a1(i1-1)+1+stride*(i2-1):a1(i1-1)+leng+stride*(i2-1),:);
            i0=i0+1;
        end
    end
end


