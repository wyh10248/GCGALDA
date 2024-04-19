import numpy as np
import torch
from Node2Vec import *
import networkx as nx

device = torch.device("cpu")
 
def constructNet(mirna_disease_matrix):
    drug_matrix = np.matrix(
        np.zeros((mirna_disease_matrix.shape[0], mirna_disease_matrix.shape[0]), dtype=np.int8))
    dis_matrix = np.matrix(
        np.zeros((mirna_disease_matrix.shape[1], mirna_disease_matrix.shape[1]), dtype=np.int8))

    mat1 = np.hstack((drug_matrix, mirna_disease_matrix))#沿着水平方向进行连接
    mat2 = np.hstack((mirna_disease_matrix.T, dis_matrix))
    adj = np.vstack((mat1, mat2))#沿着垂直方向进行连接
    return adj

def normalize(arr):
    min_val = np.min(arr)
    max_val = np.max(arr)
    
    normalized_arr = (arr + min_val) / (max_val - min_val) 
    return np.mean(normalized_arr, axis=0) 
 


def constructHNet(train_mirna_disease_matrix, mirna_matrix, disease_matrix): 
    mat1 = np.hstack((mirna_matrix, train_mirna_disease_matrix))
    mat2 = np.hstack((train_mirna_disease_matrix.T, disease_matrix))
    mat3= np.vstack((mat1, mat2))
    positive_index_tuple = np.where(mat3 > 0.5)
    positive_index_list = list(zip(positive_index_tuple[0], positive_index_tuple[1]))
    G = nx.Graph()
    for i in range(665):
        G.add_node(i)
    G.add_edges_from(positive_index_list)
    #Graph with 2708 nodes and 5278 edges

    node2vec = Node2Vec(G,
                        emb_size=665,
                        p=4,q=1, length_walk=50,
                        num_walks=10, window_size=10, num_iters=2) 

    w2v = node2vec.train(workers=4, is_loadmodel=False, is_loaddata=False)
    node_embeddings = np.zeros((665,665)) 
    #加载预训练的Word2Vec模型
    node2vec = Word2Vec.load("./data1/Node2Vec.model")

    #打开文件以保存节点向量
        # 遍历节点
    for node in range(665):
        node_id = str(node)
        vector = node2vec.wv[node]  # 获取节点向量
        node_embeddings[node] = vector
    return node_embeddings
    #return mat3

def adjacency_matrix_to_edge_index(adjacency_matrix):
  num_nodes = adjacency_matrix.shape[0]
  edge_index = torch.nonzero(adjacency_matrix, as_tuple=False).t()
  return edge_index

def train_features_choose(rel_adj_mat, features_embedding):
    rna_nums = rel_adj_mat.shape[0]
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
        for i in range(1):
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


def get_feature(A_B_feature, index, adi_matrix):
    input_features = []
    output_labels = []

    for i in range(index.shape[0]):
        A_B_index = index[i]
        feature = A_B_feature[A_B_index]
        input_features.append(feature.tolist())

        A_i, B_j = A_B_index  # Assuming A_B_index contains indices for A and B in the merged matrix
        label = adi_matrix[A_i, B_j]
        output_labels.append(label)

    input_features = np.array(input_features)
    output_labels = np.array(output_labels)

    return input_features, output_labels
