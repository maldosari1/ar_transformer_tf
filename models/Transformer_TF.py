import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Transformer_EncDec import Decoder, DecoderLayer, Encoder, EncoderLayer, ConvLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding
import numpy as np
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
import math

import numpy as np


fix_seed = 2021
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)

class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=100, default_pe=True):
        super(PositionalEmbedding, self).__init__()
        positional_enc = None
        if(default_pe):
            pe = torch.zeros(max_len, d_model).float()
            pe.require_grad = False
            position = torch.arange(0, max_len).float().unsqueeze(1)
            div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            pe = pe.unsqueeze(0)
            positional_enc = pe
        else:
            for i in [32,64]:
                pe = torch.zeros(max_len, d_model).float()
                pe.require_grad = False
                position = torch.arange(0, max_len).float().unsqueeze(1)
                div_term = (torch.arange(0, d_model, 2).float() * -(math.log(i) / d_model)).exp()
                pe[:, 0::2] = torch.sin(position * div_term)
                pe[:, 1::2] = torch.cos(position * div_term)
                pe = pe.unsqueeze(0)
                self.register_buffer('pe', pe)
                if(positional_enc == None):
                    positional_enc = pe
                else:
                    positional_enc = positional_enc + pe
        self.register_buffer('positional_enc', positional_enc)


    def forward(self, x):
        return self.positional_enc[:, :x.size(1)]

class Model(nn.Module):
    """
    Vanilla Transformer with O(L^2) complexity
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.pred_len = configs.pred_len
        self.tf_ratio = configs.tf_ratio
        self.output_attention = configs.output_attention
        self.label_len = configs.label_len
        self.c_out = configs.c_out

        # Embedding
        self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                            configs.dropout, configs.model, configs.default_pe)
        self.dec_embedding = DataEmbedding(configs.dec_in, configs.d_model, configs.embed, configs.freq,
                                           configs.dropout, configs.model)
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=configs.output_attention), configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout, output_attention=False),
                        configs.d_model, configs.n_heads),
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout, output_attention=False),
                        configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for l in range(configs.d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model),
            projection=nn.Linear(configs.d_model, configs.c_out, bias=True)
        )

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None, train=True):
        decoder_input = x_dec[:,:self.label_len,:]

        enc_out = self.enc_embedding(x_enc, x_mark_enc, enc=False)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)
        outputs = torch.zeros(x_dec.shape[0], self.pred_len, self.c_out).to(x_dec.device)
        for t in range(0, self.pred_len):

            decoder_input_7 = decoder_input
            decoder_input = self.dec_embedding(decoder_input, x_mark_dec[:,:t+self.label_len,:], enc=False)
            dec_out = self.decoder(decoder_input.float(),enc_out,x_mask=dec_self_mask, cross_mask=dec_enc_mask)
            outputs[:,t,:] = dec_out[:,-1,:]
            r = random.random()
            if r < self.tf_ratio and train is True:
                decoder_input = torch.cat((decoder_input_7, x_dec[:, t+self.label_len, :].unsqueeze(1)), 1)
            else:
                decoder_input = torch.cat((decoder_input_7, dec_out[:, -1, :].unsqueeze(1)), 1)

        if self.output_attention:
            return outputs, attns
        else:
            return outputs
