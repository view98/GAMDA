import torch
import torch.nn as nn
import torch.nn.functional as F

class TokenAttentionLayer(nn.Module):
    def __init__(self, in_features,out_features,args,concat=True):
        super(TokenAttentionLayer, self).__init__()
        self.dropout = args.dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = args.alpha
        self.concat = concat
        torch.manual_seed(args.seed)
        self.W = nn.Parameter(torch.rand(size=(in_features, out_features))) # (D, D')
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.rand(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self,h,adj):
        # h     (N,T,D)
        # adj   (N*T, N*T)
        # level (N,T)
        Wh = torch.matmul(h, self.W)  # (N,T,D) * (D, D') = (N,T, D')
        a_input = self._prepare_attentional_mechanism_input(Wh)  # (N,T,T,2D')
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(3))  # (N,T,T)

        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, Wh)   # (N,T,D')

        if self.concat:
            return self.leakyrelu(h_prime)
        else:
            return h_prime

    def _prepare_attentional_mechanism_input(self, Wh):
        with torch.no_grad():
            N = Wh.size()[1]  # number of nodes

            # Below, two matrices are created that contain embeddings in their rows in different orders.
            # (e stands for embedding)
            # These are the rows of the first matrix (Wh_repeated_in_chunks):
            # e1, e1, ..., e1,            e2, e2, ..., e2,            ..., eN, eN, ..., eN
            # '-------------' -> N times  '-------------' -> N times       '-------------' -> N times
            #
            # These are the rows of the second matrix (Wh_repeated_alternating):
            # e1, e2, ..., eN, e1, e2, ..., eN, ..., e1, e2, ..., eN
            # '----------------------------------------------------' -> N times
            #
            Wh_repeated_in_chunks = Wh.repeat_interleave(N, dim=1)
            Wh_repeated_alternating = Wh.repeat(1,N,1)

            # The all_combination_matrix, created below, will look like this (|| denotes concatenation):
            # e1 || e1
            # e1 || e2
            # e1 || e3
            # ...
            # e1 || eN
            # e2 || e1
            # e2 || e2
            # e2 || e3
            # ...
            # e2 || eN
            # ...
            # eN || e1
            # eN || e2
            # eN || e3
            # ...
            # eN || eN
            all_combinations_matrix = torch.cat([Wh_repeated_in_chunks, Wh_repeated_alternating], dim=2)
            return all_combinations_matrix.view(Wh.size()[0],N, N, 2 * self.out_features)

class EventAttentionLayer(nn.Module):
    def __init__(self, in_features,out_features,args,concat=True):
        super(EventAttentionLayer, self).__init__()
        self.dropout = args.dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = args.alpha
        self.concat = concat

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features))) # (D, D')
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.b = nn.Parameter(torch.empty(size=(args.event_nums, 1)))
        nn.init.xavier_uniform_(self.b.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self,h,adj):
        # h     (N,T,D)
        # adj   (N*T, N*T)
        # level (N,T)
        Wh = torch.matmul(h, self.W)  # (N,T,D) * (D, D') = (N,T, D')
        a_input = self._prepare_attentional_mechanism_input(Wh)  # (N,T,T,2D')
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(3))  # (N,T,T)

        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, Wh)   # (N,T,D')

        if self.concat:
            return self.leakyrelu(h_prime)
        else:
            h_prime = torch.matmul(h_prime.transpose(1, 2), self.b).squeeze(2)  # (N,D',T)*(T,1)
            return h_prime

    def _prepare_attentional_mechanism_input(self, Wh):
        with torch.no_grad():
            N = Wh.size()[1]  # number of nodes

            # Below, two matrices are created that contain embeddings in their rows in different orders.
            # (e stands for embedding)
            # These are the rows of the first matrix (Wh_repeated_in_chunks):
            # e1, e1, ..., e1,            e2, e2, ..., e2,            ..., eN, eN, ..., eN
            # '-------------' -> N times  '-------------' -> N times       '-------------' -> N times
            #
            # These are the rows of the second matrix (Wh_repeated_alternating):
            # e1, e2, ..., eN, e1, e2, ..., eN, ..., e1, e2, ..., eN
            # '----------------------------------------------------' -> N times
            #
            Wh_repeated_in_chunks = Wh.repeat_interleave(N, dim=1)
            Wh_repeated_alternating = Wh.repeat(1,N,1)

            # The all_combination_matrix, created below, will look like this (|| denotes concatenation):
            # e1 || e1
            # e1 || e2
            # e1 || e3
            # ...
            # e1 || eN
            # e2 || e1
            # e2 || e2
            # e2 || e3
            # ...
            # e2 || eN
            # ...
            # eN || e1
            # eN || e2
            # eN || e3
            # ...
            # eN || eN
            all_combinations_matrix = torch.cat([Wh_repeated_in_chunks, Wh_repeated_alternating], dim=2)
            return all_combinations_matrix.view(Wh.size()[0],N, N, 2 * self.out_features)

