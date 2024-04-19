RNA=load('RNASiNet.csv');
load('GSL.mat', 'lncrna_gsSim');
a=0.9;
matrix2 = RNA;
matrix3 = lncrna_gsSim;
RNASi_kernel=a*matrix2+(1-a)*matrix3;
save('RNASi_kernel_0.9.mat','RNASi_kernel')