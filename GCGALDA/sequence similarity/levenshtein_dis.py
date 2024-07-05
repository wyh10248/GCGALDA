from Levenshtein import distance
import numpy as np
import pandas as pd

def get_distance():
    a = pd.read_excel('data2/my_circRNA_sequece_information.xlsx', header=None, keep_default_na=False)
    RNA = a.iloc[:, 0].tolist()
    seq = a.iloc[:, 1].tolist()

    # 创建矩阵
    final_matrix = np.zeros((len(seq),len(seq)))

    # 求levenshtein
    for i in range(final_matrix.shape[0]):

        for k in range(final_matrix.shape[0]):
            if i == k:
                final_matrix[i][k] = 1
                continue

            length = len(seq[i]) + len(seq[k])
            if(length == 0):
                final_matrix[i][k]=0
            else:
                final_matrix[i][k] = 1 - (distance(seq[i], seq[k], weights=(1, 1, 2)) / (len(seq[i]) + len(seq[k])))

            # print(final_matrix[i][k])

        if i%50==0:
            print(f"第{i}已结束")

    # 存储
    np.save('data2/circRNA_sequence_similarity.npy',final_matrix)
    result = pd.DataFrame(final_matrix)
    result.to_excel('data2/circRNA_sequence_similarity.xlsx', header=0, index=0)



get_distance()