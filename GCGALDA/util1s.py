import numpy as np
import torch


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
 
def constructNet(mirna_disease_matrix):
    drug_matrix = np.matrix(
        np.zeros((mirna_disease_matrix.shape[0], mirna_disease_matrix.shape[0]), dtype=np.int8))
    dis_matrix = np.matrix(
        np.zeros((mirna_disease_matrix.shape[1], mirna_disease_matrix.shape[1]), dtype=np.int8))

    mat1 = np.hstack((drug_matrix, mirna_disease_matrix))#沿着水平方向进行连接
    mat2 = np.hstack((mirna_disease_matrix.T, dis_matrix))
    adj = np.vstack((mat1, mat2))#沿着垂直方向进行连接
    return adj

def constructHNet(train_mirna_disease_matrix, mirna_matrix, disease_matrix):
    mat1 = np.hstack((mirna_matrix, train_mirna_disease_matrix))
    mat2 = np.hstack((train_mirna_disease_matrix.T, disease_matrix))
    mat3= np.vstack((mat1, mat2))
    node_embeddings = np.loadtxt('./data1/node_embeddings_a=0.1_b=0.1.csv' ,delimiter=',')
    node_embeddings = np.hstack((node_embeddings, mat3))
    node_embeddings = torch.tensor(node_embeddings)
    return node_embeddings

def adjacency_matrix_to_edge_index(adjacency_matrix):
  num_nodes = adjacency_matrix.shape[0]
  edge_index = torch.nonzero(adjacency_matrix, as_tuple=False).t()
  return edge_index

def train_features_choose(rel_adj_mat, features_embedding, negative_sample_times):
    rna_nums = rel_adj_mat.size()[0]
    features_embedding_rna = features_embedding[0:rna_nums, :]
    features_embedding_dis = features_embedding[rna_nums:features_embedding.size()[0], :]
    train_features_input, train_lable = [], []
    # positive position index
    positive_index_tuple = torch.where(rel_adj_mat == 1)
    positive_index_list = list(zip(positive_index_tuple[0], positive_index_tuple[1]))

    for (r, d) in positive_index_list:
        # positive samples
        train_features_input.append((features_embedding_rna[r, :] * features_embedding_dis[d, :]).unsqueeze(0))
        train_lable.append(1)
        # negative samples
        negative_colindex_list = []
        for i in range(negative_sample_times):
            j = np.random.randint(rel_adj_mat.size()[1])
            while (r, j) in positive_index_list:
                j = np.random.randint(rel_adj_mat.size()[1])
            negative_colindex_list.append(j)
        for nums_1 in range(len(negative_colindex_list)):
            train_features_input.append(
                (features_embedding_rna[r, :] * features_embedding_dis[negative_colindex_list[nums_1], :]).unsqueeze(0))
        for nums_2 in range(len(negative_colindex_list)):
            train_lable.append(0)
    train_features_input = torch.cat(train_features_input, dim=0)
    train_lable = torch.FloatTensor(np.array(train_lable)).unsqueeze(1)
    return train_features_input.to(device), train_lable.to(device)


def test_features_choose(rel_adj_mat, features_embedding):
    rna_nums, dis_nums = rel_adj_mat.size()[0], rel_adj_mat.size()[1]
    features_embedding_rna = features_embedding[0:rna_nums, :]
    features_embedding_dis = features_embedding[rna_nums:features_embedding.size()[0], :]
    test_features_input, test_lable = [], []

    for i in range(rna_nums):
        for j in range(dis_nums):
            test_features_input.append((features_embedding_rna[i, :] * features_embedding_dis[j, :]).unsqueeze(0))
            test_lable.append(rel_adj_mat[i, j])

    test_features_input = torch.cat(test_features_input, dim=0)
    test_lable = torch.FloatTensor(np.array(test_lable)).unsqueeze(1)
    return test_features_input.to(device), test_lable.to(device)


def sort_matrix(score_matrix, interact_matrix):
    '''
    实现矩阵的列元素从大到小排序
    1、np.argsort(data,axis=0)表示按列从小到大排序
    2、np.argsort(data,axis=1)表示按行从小到大排序
    '''
    sort_index = np.argsort(-score_matrix, axis=0)  # 沿着行向下(每列)的元素进行排序
    score_sorted = np.zeros(score_matrix.shape)
    y_sorted = np.zeros(interact_matrix.shape)
    for i in range(interact_matrix.shape[1]):
        score_sorted[:, i] = score_matrix[:, i][sort_index[:, i]]
        y_sorted[:, i] = interact_matrix[:, i][sort_index[:, i]]
    return y_sorted, score_sorted

