import torch
import torch.nn as nn
from layers import TokenAttentionLayer,EventAttentionLayer
import torch.nn.functional as F

class Token_GAT(nn.Module):
    def __init__(self, args,in_dim,out_dim,layer_num,head_num):
        super(Token_GAT, self).__init__()

        self.gats = []
        self.dropout = nn.Dropout(args.dropout)
        for k in range(layer_num - 1):
            self.gats.append([TokenAttentionLayer(in_dim, out_dim, args,concat=True).to(args.device) for _ in
                                   range(head_num)])
            in_dim = out_dim
        self.gat_out = TokenAttentionLayer(in_dim, out_dim, args,concat=False).to(args.device)

    def forward(self, input_feature,adj):
        for gat in self.gats:
            out_feature = [g(input_feature,adj).unsqueeze(1) for g in gat]
            out_feature = torch.cat(out_feature, dim=1)
            out_feature = out_feature.mean(dim=1)
            out_feature = self.dropout(out_feature)
            input_feature = out_feature
        out_feature = F.relu(self.gat_out(input_feature,adj))
        return out_feature

class Event_GAT(nn.Module):
    def __init__(self, args,in_dim,out_dim,layer_num,head_num):
        super(Event_GAT, self).__init__()

        self.gats = []
        self.dropout = nn.Dropout(args.dropout)
        for k in range(layer_num - 1):
            self.gats.append([EventAttentionLayer(in_dim, out_dim, args,concat=True).to(args.device) for _ in
                                   range(head_num)])
            in_dim = out_dim
        self.gat_out = EventAttentionLayer(in_dim, out_dim, args,concat=False).to(args.device)

    def forward(self, input_feature,adj):
        for gat in self.gats:
            out_feature = [g(input_feature,adj).unsqueeze(1) for g in gat]
            out_feature = torch.cat(out_feature, dim=1)
            out_feature = out_feature.mean(dim=1)
            out_feature = self.dropout(out_feature)
            input_feature = out_feature
        out_feature = F.relu(self.gat_out(input_feature,adj))
        return out_feature
