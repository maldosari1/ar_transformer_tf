import torch
from torch import nn
import math

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.n_features = configs.enc_in
        self.hidden_dim = configs.hidden_dim
        self.layer_dim = configs.layer_dim
        self.pred_len = configs.pred_len
        self.seq_len = configs.seq_len

        self.lstm = nn.LSTM(self.seq_len*self.n_features, self.hidden_dim, self.layer_dim, batch_first=True)
        self.fc = nn.Linear(self.hidden_dim, self.pred_len*self.n_features)
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None, train=True):
        if(x_enc.shape[0] == 1):
            x = torch.flatten(x_enc, start_dim=1)
        else: 
            x = torch.flatten(torch.squeeze(x_enc), start_dim=1)        
        out, (hidden, cell) = self.lstm(x)
        out = self.fc(out)
        x = torch.reshape(out, (out.shape[0],self.pred_len, self.n_features))
        return x