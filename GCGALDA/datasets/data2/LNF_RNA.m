RNA=load('lncRNA-lncRNA.txt');
load('GSL.mat', 'LGS');
a=0.1;
matrix2 = RNA;
matrix3 = LGS;
RNASi_kernel=a*matrix2+(1-a)*matrix3;
save('RNASi_kernel_0.1.mat','RNASi_kernel')