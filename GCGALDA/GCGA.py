import torch.nn as nn
from torch_geometric.nn import GATConv
from torch_geometric.nn import GCNConv
import torch
import torch.nn.functional as F
from util1s import train_features_choose, test_features_choose

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class MLP(nn.Module):
    def __init__(self, embedding_size, drop_rate):
        super(MLP, self).__init__()
        self.embedding_size = embedding_size
        self.drop_rate = drop_rate

        def init_weights(m):
            if type(m) == nn.Linear:
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif type(m) == nn.Conv2d:
                
                nn.init.uniform_(m.weight)

        self.mlp_prediction = nn.Sequential(
            nn.Linear(self.embedding_size, self.embedding_size // 4), 
            nn.LeakyReLU(),
            nn.Dropout(self.drop_rate),
            nn.Linear(self.embedding_size // 4, self.embedding_size // 2),   
            nn.LeakyReLU(),
            nn.Dropout(self.drop_rate),
            nn.Linear(self.embedding_size // 2, 1, bias=False), 
            nn.Sigmoid()
        ).to(device)
        self.mlp_prediction.apply(init_weights)

    def forward(self, rd_features_embedding):
        predict_result = self.mlp_prediction(rd_features_embedding)
        return predict_result

class GCGALDA(torch.nn.Module):
    def __init__(self, negative_times, features_embedding_size,drop_rate,in_features, gconv=GATConv, n_side_features=0 ):
        super(GCGALDA, self).__init__()
        self.negative_times=negative_times
        self.features_embedding_size = features_embedding_size
        self.drop_rate=drop_rate
        self.convs = torch.nn.ModuleList()
        self.convs2= torch.nn.ModuleList()
        self.convs.append(gconv((in_features), 64,heads=8,dropout=0.2))
        self.convs2.append(GCNConv(64*8, 256))
        self.convs2.append(GCNConv(256,128))
        self.mlp_prediction = MLP(self.features_embedding_size, self.drop_rate)
       

    def forward(self,x,edge_index,rel_matrix,train_model):
        relu=nn.ReLU()
        for conv in self.convs:
            x = F.elu(conv(x, edge_index))
        for conv in self.convs2:
            x = F.elu(conv(x, edge_index))
        outputs=relu(x)
        print('features_embedding_size:', outputs.size()[1])
        
        if train_model:
           train_features_inputs, train_lable = train_features_choose(rel_matrix, outputs, self.negative_times)
           train_mlp_result = self.mlp_prediction(train_features_inputs)
           return train_mlp_result, train_lable
        else:
           test_features_inputs, test_lable = test_features_choose(rel_matrix, outputs)
           test_mlp_result = self.mlp_prediction(test_features_inputs)
           return test_mlp_result, test_lable


