function data2=generate_img(data1,range1,leng,stride)
a1=find(range1==0);
i0=1;
for i1=2:length(a1);
    if a1(i1)-a1(i1-1)-1>leng
        for i2=1:a1(i1)-a1(i1-1)-leng
            data2(:,:,i0)=data1(a1(i1-1)+1:a1(i1-1)+leng,:);
        end
    end
end


