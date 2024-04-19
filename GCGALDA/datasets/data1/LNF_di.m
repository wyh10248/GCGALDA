load('GSD.mat', 'diease_gsSim');
data=load('DiseaseSimilarityModel.csv');
b=0.2;
matrix2 = data;
matrix3 = diease_gsSim;
DiSi_kernel=b*matrix2+(1-b)*matrix3;
save('DiSi_kernel_0.2.mat','DiSi_kernel')