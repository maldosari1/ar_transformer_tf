import torch
from torch import nn
import math
import random

class LSTMEncoder(nn.Module):
    def __init__(self,n_features, hidden_dim, layer_dim, n_past):
        super(LSTMEncoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.lstm = nn.LSTM(n_features, hidden_dim, layer_dim, batch_first=True)

    def forward(self, x):
        out, (hidden, cell) = self.lstm(x)
        return hidden, cell
    
class LSTMDecoder(nn.Module):
    def __init__(self,n_features, hidden_dim, layer_dim):
        super(LSTMDecoder, self).__init__()
        self.lstm = nn.LSTM(n_features, hidden_dim, layer_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, n_features)

    def forward(self, x, hidden, cell):
        output, (hidden, cell) = self.lstm(x, (hidden, cell))
        out = self.fc(hidden[-1])
    
        return out, hidden, cell
    
class Model(nn.Module):    
    def __init__(self, configs):
        super(Model, self).__init__()
        self.n_features = configs.enc_in
        self.hidden_dim = configs.hidden_dim
        self.layer_dim = configs.layer_dim
        self.pred_len = configs.pred_len
        self.seq_len = configs.seq_len
        self.tf_ratio = configs.tf_ratio

        self.encoder = LSTMEncoder(self.n_features,self.hidden_dim, self.layer_dim, self.seq_len)
        self.decoder = LSTMDecoder(self.n_features, self.hidden_dim, self.layer_dim)
        
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None, train=True):

        src = x_enc
        trg = x_dec
        device = x_enc.device

        
        hidden, cell = self.encoder(src)


        outputs = torch.zeros(trg.shape[0], self.pred_len, self.n_features).to(device)
        input_trg = src[:,src.shape[1]-1:src.shape[1],:]
        start = 0
        end = 1

        if(train == True):
            hidden, cell = self.encoder(src)
            
            for t in range(0, self.pred_len):
                out, hidden, cell = self.decoder(input_trg, hidden, cell)
                outputs[:,t,:] = out
                if random.random() < self.tf_ratio:
                    input_trg = trg[:,start:end,:]
                else:
                    input_trg = out.unsqueeze(1)
                start = end 
                end = end + 1

        elif(train == False):
            for t in range(0, self.pred_len):
                out, hidden, cell = self.decoder(input_trg, hidden, cell)
                outputs[:,t,:] = out
                input_trg = out.unsqueeze(1)
                start = end 
                end = end + 1
        return outputs