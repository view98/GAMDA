import torch
import torch.nn as nn
from GAT import Token_GAT,Event_GAT
from Conversion import Node_Conversion

torch.set_printoptions(profile="full")


class SG_EEL(nn.Module):
    def __init__(self, args):
        super(SG_EEL, self).__init__()
        self.args = args

        if args.embedding_type == 'glove':
            num_embeddings, embed_dim = args.glove_embedding.shape
            self.embed = nn.Embedding(num_embeddings, embed_dim)
            self.embed.weight = nn.Parameter(args.glove_embedding, requires_grad=False)
        else:
            num_embeddings, embed_dim = args.word2vec_embedding.shape
            self.embed = nn.Embedding(num_embeddings, embed_dim)
            self.embed.weight = nn.Parameter(args.word2vec_embedding, requires_grad=False)

        self.dropout = nn.Dropout(args.dropout)

        self.bilstm = nn.LSTM(input_size=args.token_embedding_dim, hidden_size=args.hidden_size,
                              bidirectional=True, batch_first=True, num_layers=args.num_layers)

        in_dim =  2 * args.hidden_size
        out_dim = 2 * args.hidden_size
        args.token_gat_layer_num = 1
        self.token_gat = Token_GAT(args,in_dim,out_dim,args.token_gat_layer_num,args.num_heads)

        self.phrase_coarsening = Node_Conversion(in_dim,out_dim,args,args.phrase_nums)
        self.phrase_gat = Token_GAT(args, in_dim,out_dim ,args.token_gat_layer_num,args.num_heads)

        self.structure_coarsening = Node_Conversion(in_dim,out_dim,args,args.structure_nums)
        self.structure_gat = Token_GAT(args, in_dim, out_dim, args.token_gat_layer_num, args.num_heads)

        self.event_coarsening = Node_Conversion(in_dim,out_dim,args,1)
        in_dim = 2*args.hidden_size + args.token_embedding_dim
        out_dim =in_dim
        self.event_gat = Event_GAT(args, in_dim, out_dim, args.token_gat_layer_num, args.num_heads)

        torch.manual_seed(args.seed)
        self.eW = nn.Parameter(torch.rand(size=(1, args.event_nums)))  # (1,Q)
        nn.init.xavier_uniform_(self.eW.data, gain=1.414)
        self.pool2d = nn.MaxPool2d((self.args.token_nums, 1))

        last_hidden_size = 2*args.hidden_size + args.token_embedding_dim

        layers = [nn.Linear(last_hidden_size, args.final_hidden_size), nn.LeakyReLU()]
        for _ in range(args.num_mlps - 1):
            layers += [nn.Linear(args.final_hidden_size,
                                 args.final_hidden_size), nn.LeakyReLU()]
        self.fcs = nn.Sequential(*layers)
        self.fc_final = nn.Linear(args.final_hidden_size, args.num_classes)

    def forward(self,token_ids,event_ids,token_adj,phrase_adj,structure_adj,event_adj,token2phrase,phrase2structure,structure2event):
        token_feature = self.embed(token_ids)  # (N,T,F)
        token_feature = self.dropout(token_feature)

        token_out_bilstm, _ = self.bilstm(token_feature)  # (N,T,2D)
        token_out_bilstm = self.dropout(token_out_bilstm)
        token_out_gat = self.token_gat(token_out_bilstm,token_adj)  # (N,T,2D)
        token_out_gat = self.dropout(token_out_gat)
        phrase_in = self.phrase_coarsening(token_out_gat,token2phrase) # (N,M,2D)
        phrase_in = self.dropout(phrase_in)

        out_phrase_gat = self.phrase_gat(phrase_in,phrase_adj) # (N,M,2D)
        out_phrase_gat = self.dropout(out_phrase_gat)
        structure_in = self.structure_coarsening(out_phrase_gat,phrase2structure) # (N,P,2D)
        structure_in = self.dropout(structure_in)

        out_structure_gat = self.structure_gat(structure_in,structure_adj) # (N,P,2D)
        out_structure_gat = self.dropout(out_structure_gat)
        event_in = self.event_coarsening(out_structure_gat,structure2event) # (N,1,2D)
        event_in = self.dropout(event_in)

        event_feature = self.embed(event_ids)  # (N, Q,T,D')
        event_feature = self.dropout(event_feature)

        event_out_pooling = self.pool2d(event_feature).squeeze(2)   # (N,Q,1,D)->(N,Q,D')
        event_out_pooling = self.dropout(event_out_pooling)

        event_in = torch.matmul(event_in.transpose(1,2),self.eW)  # (N,2D,1)*(1,Q)=(N,2D,Q)
        event_in = event_in.transpose(1,2) # (N, Q, 2D)
        event_in = torch.cat([event_in,event_out_pooling],dim=2)  # (N,Q,2D+F)
        event_in = self.dropout(event_in)

        out_event_gat = self.event_gat(event_in,event_adj) # (N,2D+D')
        out_event_gat = self.dropout(out_event_gat) # (N,2D+D')

        out = self.fcs(out_event_gat)
        logit = self.fc_final(out)  # (N,10)
        return logit





