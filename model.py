import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
import math
from torch.autograd import Variable

#using attention
class Transformer(nn.Module):
    def __init__(self, encoder, decoder):
        super(Transformer,self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self,x, z):
        c = self.encoder(x)
        y = self.decoder(z,c)
        return y

class Encoder(nn.Module):
    def __init__(self, encoder_layer, n_layer):
        super(Encoder, self).__init__()
        self.layer = []
        for i in range(n_layer):
            self.layer.append(copy.deepcopy(encoder_layer))

    def foward(self, x):
        out = x
        for layer in self.layers:
            out = layer(out)
        return out

class EncoderLayer(nn.Module):
    def __init__(self, multi_head_attention_layer, position_wise_feed_forward_layer, norm_layer):
        super(EncoderLayer, self).__init__()
        self.multi_head_attention_layer = multi_head_attention_layer
        self.position_wise_feed_forward_layer = position_wise_feed_forward_layer
        self.residual_connection_layers = [ResidualConnectionLayer(copy.deepcopy(norm_layer)) for i in range(2)]

    def foward(self, x, mask):
        out = self.residual_connection_layers[0](x, lambda x: self.multi_head_attention_layer(x, x, x, mask))
        out = self.residual_connection_layers[1](x, lambda x: self.position_wise_feed_forward_layer(x))
        return out


class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, d_model, h, qkv_fc_layer, fc_layer):
        super(MultiHeadAttentionLayer, self).__init__()
        self.d_model = d_model
        self.h = h
        self.query_fc_layer = copy.deepcopy(qkv_fc_layer)
        self.key_fc_layer = copy.deepcopy(qkv_fc_layer)
        self.value_fc_layer = copy.deepcopy(qkv_fc_layer)
        self.fc_layer = fc_layer
    
    def calculate_attention(self, query, key, value, mask):
        # query, key, value's shape: (n_batch, seq_len, d_k)
        d_k = key.size(-1) # get d_k
        attention_score = torch.matmul(query, key.transpose(-2,-1)) # Q * K^T, attention_score'shape : (n_batch, seq_len, seq_len)
        attention_score = attention_score / math.sqrt(d_k) # scaling
        if mask is not None:
            attention_score = attention_score.masked_fill(mask==0, -1e9) # masking

        attention_prob = F.softmax(attention_score, dim=-1) # softmax, attention_prob's shape: (n_batch, seq_len, seq_len)
        out = torch.matmul(attention_prob, value) # Attention_Prob x V, out's shape: (n_batch, seq_len, d_k)
        return out


    def foward(self, query, key, value, mask =None):
        n_batch = query.shape[0] # num_batch
    
        def transform(x, fc_layer): # (n_batch,seq_len, d_embed) -> (n_batch, h, seq_len, d_k)
            out = fc_layer(x) # (n_batch, seq_len, d_model)
            out = out.view(n_batch, -1, self.h, self.d_model//self.h)
            out = out.transpose(1,2)
            return out
        
        query = transform(query, self.query_fc_layer)
        key = transform(key, self.key_fc_layer)
        value = transform(value, self.value_fc_layer)

        if mask is not None:
            mask = mask.unsqueeze(1) # mask's shape : (n_batch, 1, seq_len, seq_len)
        
        out = self.calculate_attention(query, key, value, mask) # (n_batch, h, seq_len, d_k)
        out = out.transpose(1,2) # (n_batch, seq_len, h, d_k)
        out = out.contiguous().view(n_batch, -1, self.d_model) # (n_batch, seq_len, d_model)
        out = self.fc_layer(out) # (n_batch, seq_len, d_embed)
        return out

class PositionWiseFeedForwardLayer(nn.Module):
    def __init__(self, first_fc_layer, second_fc_layer):
        self.first_fc_layer = first_fc_layer
        self.second_fc_layer = second_fc_layer
    
    def foward(self, x):
        out = self.first_fc_layer(x)
        out = F.relu(out)
        out = F.dropout(out)
        out = self.second_fc_layer(out)
        return out

class ResidualConnectionLayer(nn.Module):
    def __init__(self,norm_layer):
        super(ResidualConnectionLayer, self).__init__()
        self.norm_layer = norm_layer
    
    def forward(self, x, sub_layer):
        out = sub_layer(x) + x
        out = self.norm_layer(out)
        return out

def subsequent_mask(size):
    attention_shape = (1, size, size)
    mask = np.triu(np.ones(attention_shape), k = 1).astype('uint8') # masking with upper triangle matrix
    return torch.from_numpy(mask)==0

def make_std_mask(target, padding):
    target_mask = (target != padding)
    target_mask = target_mask.unsqueeze(-2) # (n_batch, seq_len) -> (n_batch, 1, seq_len)
    target_mask = target_mask & Variable(subsequent_mask(target.size(-1))).type_as(target_mask.data)
    return target_mask

