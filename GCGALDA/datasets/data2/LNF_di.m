load('GSD.mat', 'DGS');
data=load('disease-disease.txt');
b=0.9;
matrix2 = data;
matrix3 = DGS;
DiSi_kernel=b*matrix2+(1-b)*matrix3;
save('DiSi_kernel_0.9.mat','DiSi_kernel')