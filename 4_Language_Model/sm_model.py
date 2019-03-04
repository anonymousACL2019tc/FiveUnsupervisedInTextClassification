import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.nn import functional as F

from embed_regularize import embedded_dropout
from locked_dropout import LockedDropout
from weight_drop import WeightDrop


class LMmodel(nn.Module):
    """
    Container module with an encoder, a recurrent module, and a decoder.
    Modified by Sung-Min, Yang. 18th, Febrary, 2018.
    Original code : https://github.com/salesforce/awd-lstm-lm
    """

    def __init__(self, rnn_type, ntoken, ninp, nhid, nlayers, 
                 dropout=0.5, dropouth=0.5, dropouti=0.5, dropoute=0.1, wdrop=0, tie_weights=False,
                 pooling=False):
        super(LMmodel, self).__init__()
        
        self.lockdrop = LockedDropout()
        
        self.ntoken = ntoken # <---------------- Temporary, probably <NUM>, <MIX_NUM> in another dataset.
        self.idrop = nn.Dropout(dropouti)
        self.hdrop = nn.Dropout(dropouth)
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(self.ntoken, ninp)
        
        # Pre-trained model doens't use batch_first.
        if rnn_type == 'LSTM':
            self.rnns = [torch.nn.LSTM(ninp if l == 0 else nhid, nhid if l != nlayers - 1 else (ninp if tie_weights else nhid), 1, dropout=0) for l in range(nlayers)]
            if wdrop:
                self.rnns = [WeightDrop(rnn, ['weight_hh_l0'], dropout=wdrop) for rnn in self.rnns]
        print(self.rnns)
        self.rnns = torch.nn.ModuleList(self.rnns)


        self.rnn_type = rnn_type
        self.ninp = ninp
        self.nhid = nhid
        self.nlayers = nlayers
        self.dropout = dropout
        self.dropouti = dropouti
        self.dropouth = dropouth
        self.dropoute = dropoute
        self.tie_weights = tie_weights
        self.pooling = pooling

    

    def forward(self, input, seq_length, hidden=False):
        
        
        x_emb = embedded_dropout(self.encoder, input, dropout=self.dropoute if self.training else 0)
        X_emb = self.lockdrop(x_emb, self.dropouti)
        
        
        # Unpack template
        # packed_input = pack_padded_sequence(x_emb, seq_length)
        # packed_output, (ht, ct) = self.lstm(packed_input)
        # output, _ = pad_packed_sequence(packed_output)
        
        new_hidden = []
        outputs = []
        ht_list = []
        
        seq_length = seq_length.cpu().numpy() # Make it to numpy array type.
        
        for i, rnn in enumerate(self.rnns):
            packed_input = pack_padded_sequence(x_emb, seq_length)    # Pack
            if hidden:
                packed_output, (ht, ct) = rnn(packed_input, hidden[i])
            elif not hidden:
                packed_output, (ht, ct) = rnn(packed_input)
            unpacked_output, _ = pad_packed_sequence(packed_output)    # Unpack
            
            current_hidden_states = ht
            new_hidden.append(ht) # Not save cell-states
            outputs.append(unpacked_output)
            
            if i != self.nlayers-1 :
                x_emb = self.lockdrop(unpacked_output, self.dropouth)  # unpacked_output as a input of next lstm layer.
            # if self.relu: x_emb = F.relu(x_emb)
            
        
        # 1. Select last hidden state, 
        #  Did not implement locked drop when we are obtaining last hidden state.
        # X -> last_hidden_state = self.lockdrop(ht[-1], self.dropout)
        
        last_hidden_400 = ht[-1]  # <------------------------------------------- CORRECT
        
        if self.pooling == "last_hidden":
            # last_hidden_400
            # @TODO
            # implement -> pool_4_nlp() function
            concat_input = last_hidden_400
        
        elif self.pooling == "smart_pool":
            avg_400 = pool_4_nlp(outputs[1], pool_type="average", batch_first=False, window_size=5, pad=0, nunit=400, stride=1)
            # max_400 = pool_4_nlp(outputs[0], pool_type="max", batch_first=False, window_size=5, pad=0, nunit=200, stride=1)
            # import pdb;pdb.set_trace()
            concat_input = torch.cat([last_hidden_400, avg_400],1)
        
        elif self.pooling == "multi_hidden_pool":
            hidden_states = [ht[-1] for ht in new_hidden]
            concat_input = torch.cat(hidden_states,1)
            
        elif self.pooling == "mixed_pool":
            
            avg_400_first = torch.mean(outputs[0], 0)
            avg_400_second = torch.mean(outputs[1], 0)
#             max_400_first = torch.max(outputs[0], 0)[0]
            avg_400_first = F.adaptive_avg_pool1d(avg_400_first.unsqueeze(0), 400).squeeze()
            avg_400_second = F.adaptive_avg_pool1d(avg_400_second.unsqueeze(0), 400).squeeze()
#             max_400_first = F.adaptive_max_pool1d(max_400_first.unsqueeze(0), 400).squeeze()
            avg_400_last = torch.mean(outputs[-1], 0)
            max_400_last = torch.max(outputs[-1], 0)[0]
            
#             concat_input = torch.cat([last_hidden_400, avg_400_first, max_400_first, avg_400_last, max_400_last],1)
            if len(last_hidden_400) == 1: last_hidden_400 = last_hidden_400[0] # in case, user is using batch size 1
        
            concat_input = torch.cat([last_hidden_400, avg_400_first, avg_400_second, avg_400_last, max_400_last],1)
            
        else:
            # 2. Get average, max value of last layers
            avg_400 = torch.mean(unpacked_output, 0)
            max_400 = torch.max(unpacked_output, 0)[0]

            concat_input = torch.cat([last_hidden_400, avg_400, max_400],1)
        
        return concat_input, new_hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        if self.rnn_type == 'LSTM':
            return [(Variable(weight.new(1, bsz, self.nhid if l != self.nlayers - 1 else (self.ninp if self.tie_weights else self.nhid)).zero_()),
                    Variable(weight.new(1, bsz, self.nhid if l != self.nlayers - 1 else (self.ninp if self.tie_weights else self.nhid)).zero_()))
                    for l in range(self.nlayers)]
                    

class ClassifierModel(nn.Module):
    """
    Simple fully cunnected layers module.
    """


    def __init__(self, input_dim, nclass, nhid_linear, batch_norm, dropout_additional, relu):
        super(ClassifierModel, self).__init__()
        
        if dropout_additional: self.dropout_addtional_layer = nn.Dropout(p=dropout_additional)
        if batch_norm: self.batch_normalize = nn.BatchNorm1d(input_dim)
        
        self.classifier1 = nn.Linear(input_dim, nhid_linear)
        self.classifier2 = nn.Linear(nhid_linear, nclass)
        
        self.batch_norm = batch_norm
        self.dropout_additional = dropout_additional
        self.relu= relu


            
    def forward(self, concat_input):
        if self.dropout_additional: concat_input = self.dropout_addtional_layer(concat_input)
        if self.relu:               concat_input = F.relu(concat_input)
        if self.batch_normalize:    concat_input = self.batch_normalize(concat_input)
        
        result = self.classifier1(concat_input) # <-------------------- Either use, last hidden or avg
        result = self.classifier2(result)
        return result