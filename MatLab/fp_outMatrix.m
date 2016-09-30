function fp_outMatrix(f, name)

[cols, rows] = size(f);

% write the image to disk
filename = name; 
fileID = fopen(filename,'wt');
fprintf(fileID, '%d,%d\n', rows,cols);
for i=1:rows
    for j=1:cols
        fprintf(fileID, '%e,%e\n', real(f(i,j)),imag(f(i,j)));
    end
end
fclose(fileID);