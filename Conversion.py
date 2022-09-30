import torch
import torch.nn as nn
import torch.nn.functional as F

class Node_Conversion(nn.Module):
    def __init__(self,in_features,out_features,args,coarsen_nums):
        super(Node_Conversion, self).__init__()
        self.dropout = args.dropout
        self.alpha = args.alpha
        self.in_features = in_features
        self.out_features = out_features
        torch.manual_seed(args.seed)
        self.W = nn.Parameter(torch.rand((in_features, out_features)))  # (D, D')
        nn.init.xavier_uniform_(self.W.data, gain=1.414)

        self.a = nn.Parameter(torch.rand((out_features,coarsen_nums)))  # (D', M)
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.b = nn.Parameter(torch.rand((coarsen_nums, coarsen_nums)))  # (M, M)
        nn.init.xavier_uniform_(self.b.data, gain=1.414)

        self.eye = torch.eye(coarsen_nums,dtype=torch.int32).to(args.device)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, features,conversion_adj):
        # features (N, T, D')
        # conversion_adj (N, T, M)

        Wh = torch.matmul(features, self.W)  # (N,T,D) * (D, D') = (N,T, D')
        e = self.leakyrelu(torch.matmul(Wh, self.a))  # (N,T, D') * (D',M) = (N,T,M)

        zero_vec = -9e15 * torch.ones_like(e)
        token_attention = torch.where(conversion_adj > 0, e, zero_vec) # (N,T,M)
        token_attention = F.softmax(token_attention,dim=1)
        token_attention = F.dropout(token_attention, self.dropout, training=self.training)

        phrase_zero_vec = -9e15 * torch.ones_like(self.b)
        phrase_attention = torch.where(self.eye>0,self.b,phrase_zero_vec)  # (M,M)
        phrase_attention = F.softmax(phrase_attention,dim=1)
        phrase_attention = F.dropout(phrase_attention, self.dropout, training=self.training)

        attention = torch.matmul(token_attention,phrase_attention).transpose(1,2)  # (N,T,M)* (M,M)= (N,M,T)
        attention = F.softmax(attention,dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)

        conversion_feature = torch.matmul(attention, Wh) # (N,M,T) * (N,T,D') = (N,M,D')
        conversion_feature = F.dropout(conversion_feature,self.dropout,training=self.training)

        return F.relu(conversion_feature)
