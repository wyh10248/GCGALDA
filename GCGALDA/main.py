import argparse
import nni
import math
import random
from scipy.io import loadmat
from util1s import *
import numpy as np
import torch
import torch.nn.functional as F
from GCGA import GCGALDA
from util1s import  sort_matrix
import matplotlib.pyplot as plt
# 参数设置 
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--seed', type=int, default=2022, help='Random seed.')
parser.add_argument('--epochs', type=int, default=5,                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.0001,
                    help='Learning rate.')
parser.add_argument('--weight_decay', type=float, default=1e-7, 
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=64,
                    help='Dimension of representations')


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    A = np.loadtxt('./data/DiseaseAndRNABinary.csv' ,delimiter=',')
    A=A.T
    results = {} 
    disSimi1 = loadmat('./data/DiSi_kernel_0.8.mat') 
    #disSimi1 = loadmat('./data/DiSi_kernel_{:.1f}.mat'.format(value1))
    disSimi = disSimi1['DiSi_kernel']
#for m in range(1, 10):
    #value2 = m / 10.0
    RNASimi1 = loadmat('./data/RNASi_kernel_0.2')
    #RNASimi1 = loadmat('./data/RNASi_kernel_{:.1f}.mat'.format(value2))
    lncSimi = RNASimi1['RNASi_kernel']
    lncSimi_mat = torch.from_numpy(lncSimi).to(torch.float32)
    disSimi_mat = torch.from_numpy(disSimi).to(torch.float32)

    lncrna_disease_matrix = np.copy(A)
    rna_numbers = lncrna_disease_matrix.shape[0]
    dis_number = lncrna_disease_matrix.shape[1]

    # 寻找正样本的索引
    positive_index_tuple = np.where(lncrna_disease_matrix == 1)
    positive_index_list = list(zip(positive_index_tuple[0], positive_index_tuple[1]))

    # 随机打乱
    random.shuffle(positive_index_list)
    #将正样本分为5个数量相等的部分
    positive_split = math.ceil(len(positive_index_list) / 5)
    all_tpr = []
    all_fpr = []
    all_recall = []
    all_precision = []
    all_accuracy = []
    all_F1 = []
    tpr_list = []
    fpr_list = []
    recall_list = []
    precision_list = []
    accuracy_list = []
    F1_list = []
    prediction_matrix_F=[]
    count = 0
    # 5-fold
    print('starting fivefold cross validation..................')
    for i in range(0, len(positive_index_list), positive_split):
        count = count + 1
        print("This is {} fold cross validation".format(count))
        positive_train_index_to_zero = positive_index_list[i: i + positive_split]
        new_lncrna_disease_matrix = lncrna_disease_matrix.copy()
        # 五分之一的正样本置为0
        for index in positive_train_index_to_zero:
            new_lncrna_disease_matrix[index[0], index[1]] = 0
        new_lncrna_disease_matrix_tensor = torch.from_numpy(new_lncrna_disease_matrix).to(device)
        roc_lncrna_disease_matrix = new_lncrna_disease_matrix + lncrna_disease_matrix
        x=torch.tensor(constructHNet(new_lncrna_disease_matrix, lncSimi, disSimi)).to(torch.float32) 
        adj=torch.tensor(constructNet(new_lncrna_disease_matrix))
        edge_index=adjacency_matrix_to_edge_index(adj)
        lncSimi_mat = lncSimi_mat.to(device)
        disSimi_mat = disSimi_mat.to(device)
        model = GCGALDA(1,128,0.001,1330, n_side_features=665) 
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        
        # 模型训练
        model.train()
        for epoch in range(args.epochs):
            train_predict_result, train_lable = model(x,edge_index, new_lncrna_disease_matrix_tensor,
                                                      train_model=True)
            loss = F.binary_cross_entropy(train_predict_result, train_lable)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print('Epoch %d | train Loss: %.4f' % (epoch + 1, loss.item()))
        # 模型评估
        model.eval()
        with torch.no_grad():
            test_predict_result, test_lable = model(x,edge_index, new_lncrna_disease_matrix_tensor, train_model=False)
            
        prediction_matrix = np.zeros(lncrna_disease_matrix.shape)
        for num in range(test_predict_result.size()[0]):
            row_num = num // dis_number #240
            col_num = num % dis_number
            prediction_matrix[row_num, col_num] = test_predict_result[num, 0]
        zero_matrix = np.zeros(prediction_matrix.shape).astype('int64')
        prediction_matrix_temp = prediction_matrix.copy()
        prediction_matrix_temp = prediction_matrix_temp + zero_matrix
        prediction_matrix_F.append(prediction_matrix)
        min_value = np.min(prediction_matrix_temp)
        index_where_2 = np.where(roc_lncrna_disease_matrix == 2)
        # # 使参数训练的正样本得分在排序的时候下沉(从大到小排序)
        prediction_matrix_temp[index_where_2] = min_value - 20
        # 得分排序(得分矩阵排序以及对应的关联矩阵排序)
        sorted_rna_dis_matrix, sorted_prediction_matrix =sort_matrix(prediction_matrix_temp, roc_lncrna_disease_matrix)
        tpr_list = []
        fpr_list = []
        recall_list = []
        precision_list = []
        accuracy_list = []
        F1_list = []
        for cutoff in range(sorted_prediction_matrix.shape[0]):
            P_matrix = sorted_prediction_matrix[0:cutoff + 1, :]
            N_matrix = sorted_prediction_matrix[cutoff  +1:sorted_rna_dis_matrix.shape[0] + 1, :]
            TP = np.sum(P_matrix >= 0.5)
            FP = np.sum(P_matrix < 0.5)
            TN = np.sum(N_matrix < 0.5)
            FN = np.sum(N_matrix >= 0.5)
            tpr = TP / (TP + FN)
            fpr = FP / (FP + TN)
            tpr_list.append(tpr)
            fpr_list.append(fpr)
            recall = TP / (TP + FN)
            precision = TP / (TP + FP)
            recall_list.append(recall)
            precision_list.append(precision)
            accuracy = (TN + TP) / (TN + TP + FN + FP)
            F1 = (2 * TP) / (2 * TP + FP + FN)
            F1_list.append(F1)
            accuracy_list.append(accuracy)
        top_list = [10, 20, 50, 100, 200]
        for num in top_list:
            P_matrix = sorted_prediction_matrix[0:num, :]
            N_matrix = sorted_prediction_matrix[num:sorted_prediction_matrix.shape[0] + 1, :]
            top_count = np.sum(P_matrix >= 0.5)
            #print("top" + str(num) + ": " + str(top_count))
            all_tpr.append(tpr_list)
            all_fpr.append(fpr_list)
            all_recall.append(recall_list)
            all_precision.append(precision_list)
            all_accuracy.append(accuracy_list)
            all_F1.append(F1_list)
    tpr_arr = np.array(all_tpr)
    fpr_arr = np.array(all_fpr)
    recall_arr = np.array(all_recall)
    precision_arr = np.array(all_precision)
    accuracy_arr = np.array(all_accuracy)
    F1_arr = np.array(all_F1)
    np.savetxt('tpr_arr_mean_att.csv', tpr_arr, delimiter=',')
    np.savetxt("fpr_arr_mean_att.csv", fpr_arr, delimiter=',')
    np.savetxt('recall_arr_mean_att.csv', recall_arr, delimiter=',')
    np.savetxt("precision_arr_mean_att.csv", precision_arr, delimiter=',')

    mean_cross_tpr = np.mean(tpr_arr, axis=0)  # axis=0
    mean_cross_fpr = np.mean(fpr_arr, axis=0)
    mean_cross_recall = np.mean(recall_arr, axis=0)
    mean_cross_precision = np.mean(precision_arr, axis=0)
    mean_cross_accuracy = np.mean(accuracy_arr, axis=0)

    np.savetxt('mean_cross_tpr_mean_att.csv', mean_cross_tpr, delimiter=',')
    np.savetxt('mean_cross_fpr_mean_att.csv', mean_cross_fpr, delimiter=',')
    np.savetxt('mean_cross_recall_mean_att.csv', mean_cross_recall, delimiter=',')
    np.savetxt('mean_cross_precision_mean_att.csv', mean_cross_precision, delimiter=',')

    # 计算此次五折的平均评价指标数值
    mean_accuracy = np.mean(np.mean(accuracy_arr, axis=1), axis=0)
    mean_accuracy1 = np.mean(accuracy_arr)
    mean_recall = np.mean(np.mean(recall_arr, axis=1), axis=0)
    mean_precision = np.mean(np.mean(precision_arr, axis=1), axis=0)
    mean_F1 = np.mean(np.mean(F1_arr, axis=1), axis=0)
    mean_predict_matrix=np.mean(prediction_matrix_F, axis=0)
    mean_predict_matrix = np.round(mean_predict_matrix, decimals=4)
    #np.savetxt('mean_predict_matrix.csv', mean_predict_matrix, delimiter=',')
    print("accuracy:%.4f,recall:%.4f,precision:%.4f,F1:%.4f" % (mean_accuracy, mean_recall, mean_precision, mean_F1))

    roc_auc = np.trapz(mean_cross_tpr, mean_cross_fpr)
    AUPR = np.trapz(mean_cross_precision, mean_cross_recall)
    print("AUC:%.4f,AUPR:%.4f" % (roc_auc, AUPR))
    
    plt.plot(mean_cross_fpr, mean_cross_tpr, label='mean ROC=%0.4f' % roc_auc)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc=0)
    plt.show()
    plt.plot(mean_cross_recall, mean_cross_precision, label='mean AUPR=%0.4f' %  AUPR)
    plt.xlabel('recall')
    plt.ylabel('precision')
    plt.legend(loc=0)
    plt.show()