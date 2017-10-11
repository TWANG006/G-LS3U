function ToVideo(startNum, endNum)

for k=startNum:endNum
    
    k0=k-startNum+1;
    
    imgName = ['../Experiments/imgs/2/' num2str(k+1000) '.bmp'];
    I = imread(imgName);
    iResultName = ['../Experiments/imgs/2/CUDA/' num2str(k) '.bmp'];
    R = imread(iResultName);
    
    C = cat(2,I,R);
    t1(:,:,1) = C;t1(:,:,2)=C;t1(:,:,3)=C;
    dpw(k0)=im2frame(t1);
end
v = VideoWriter('newfile.avi');
open(v);
writeVideo(v,dpw);