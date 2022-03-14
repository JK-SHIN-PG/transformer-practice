from matplotlib.pyplot import xlim
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
import math
from torch.autograd import Variable

class PositionalEncoding(nn.Module):
    def __init__(self, h_dim, num_position):
        super(PositionalEncoding,self).__init__()

        self.register_buffer('pos_table', self._get_sinusoid_encoding_table(h_dim, num_position))
    
    def _get_sinusoid_encoding_table(self, h_dim, num_position):

        def get_position_angle_vel(position):
            return [position / torch.Tensor([10000]).pow(2 * (hid_j // 2) / h_dim) for hid_j in range(h_dim)]
        
        sinusoid_table = torch.Tensor([get_position_angle_vel(pos_i) for pos_i in range(num_position)])
        sinusoid_table[:, 0::2] = torch.sin(sinusoid_table[:,0::2])
        sinusoid_table[:, 1::2] = torch.cos(sinusoid_table[:,1::2])

        return torch.FloatTensor(sinusoid_table).unsqueeze(0)
    
    def forward(self, x):
        return x + self.pos_table[:, :x.size(1)].clone().detach()


class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, h_dim, num_heads, dropout_rate, device):
        super(MultiHeadAttentionLayer, self).__init__()

        assert h_dim % num_heads == 0

        self.h_dim = h_dim # Dimension of embedding
        self.num_heads = num_heads # Number of heads
        self.head_dim = h_dim // num_heads # Dimensionf of head's embedding

        self.query_fc = nn.Linear(h_dim, h_dim) # Fully connected layer for Query
        self.key_fc = nn.Linear(h_dim, h_dim) # Fully connected layer for key
        self.value_fc = nn.Linear(h_dim, h_dim) # Fully connected layer for value
        
        self.out_fc = nn.Linear(h_dim,h_dim) # Fully connected layer for output
        self.dropout = nn.Dropout(dropout_rate)
        self.device = device
    
    def calculate_attention_score(self, query, key, value, mask = None):
        # query : [batch_size, num_heads, seq_len, head_dim]
        # key.permute : [batch_size, num_heads, head_dim, seq_len]
        # attn_score : [batch_size, num_heads, seq_len, seq_len]
        
        # 1. dot-product to compute similarity
        attn_energy = torch.matmul(query, key.permute(0,1,3,2))

        # 2. scaling
        scale = torch.sqrt(torch.FloatTensor([self.h_dim])).to(self.device)
        scaled_attn_energy = attn_energy / scale

        # 3. masking (opt.)
        if mask is not None:
            scaled_attn_energy = scaled_attn_energy.masked_fill(mask==0, -1e9)

        # 4. soft-max
        attn_score = torch.softmax(scaled_attn_energy, dim = -1)

        # 5. matmul
        # scaled_dot_product_attention : [batch_size, num_heads, seq_len, head_dim]
        scaled_dot_product_attention = torch.matmul(self.dropout(attn_score), value)

        return scaled_dot_product_attention

    def forward(self, query, key, value, mask = None):
        batch_size = query.shape[0]

        # 1. pass each fully connected layer
        # all elements (Query, Key, Value) : [batch_size, seq_len, h_dim]
        Query = self.query_fc(query)
        Key = self.query_fc(key)
        Value = self.query_fc(value)

        # spliting the embedding's dimension : h_dim = self.num_heads * head_dim
        # view : [batch_size, seq_len, h_dim] -> [batch_size, seq_len, num_heads, head_dim]
        # permute : [batch_size, seq_len, num_heads, head_dim] -> [batch_size, num_heads, seq_len, head_dim]
        Query = Query.view(batch_size, -1, self.num_heads, self.head_dim).permute(0,2,1,3)
        Key = Key.view(batch_size, -1, self.num_heads, self.head_dim).permute(0,2,1,3)
        Value = Value.view(batch_size, -1, self.num_heads, self.head_dim).permute(0,2,1,3)

        # 2. calculate scaled dot product attention score
        # attn_score : [batch_size, num_heads, seq_len, head_dim]
        attn_score = self.calculate_attention_score(Query, Key, Value, mask)

        # Matching Dimension
        # [batch_size, num_heads, seq_len, head_dim] -> [batch_size, seq_len, num_heads, head_dim]
        attn_score = attn_score.transpose(1,2).contiguous()

        # 3. Concatenate 
        # attn_score : [batch_size, seq_len, num_heads, head_dim] -> [batch_size, seq_len, h_dim]
        attn_score = attn_score.view(batch_size, -1, self.h_dim)

        # 4. pass fully connected layer for output
        out = self.out_fc(attn_score)

        return out, attn_score

class PositionWiseFeedForwardLayer(nn.Module):
    def __init__(self, h_dim, inter_dim, dropout_rate):
        super(PositionWiseFeedForwardLayer,self).__init__()
        self.fc_1 = nn.Linear(h_dim, inter_dim)
        self.fc_2 = nn.Linear(inter_dim, h_dim)
        self.dropout = nn.Dropout(dropout_rate)
    def forward(self, x):
        out = self.dropout(F.relu(self.fc_1(x)))
        out = self.fc_2(out)
        return out


class EncoderLayer(nn.Module):
    def __init__(self,h_dim, num_heads, inter_dim, dropout_rate, device):
        super(EncoderLayer,self).__init__()
        self.norm = nn.LayerNorm(h_dim)
        self.ff_norm = nn.LayerNorm(h_dim)
        self.attention_layer = MultiHeadAttentionLayer(h_dim, num_heads, dropout_rate, device)
        self.feedforward_layer = PositionWiseFeedForwardLayer(h_dim, inter_dim, dropout_rate)
        self.dropout = nn.Dropout(dropout_rate)
    
    def forward(self, x, x_masking):
        # x : [batch_size, seq_len, h_dim]
        # x_masking : [batch_size, seq_len]

        # 1. Multi-Head Attention
        _x, _ = self.attention_layer(x, x, x, x_masking)
        
        # 2. Add & norm
        x = self.norm(x + self.dropout(_x))

        # 3. Positional Feed Forward
        _x = self.feedforward_layer(x)

        # 4. Add & norm
        x = self.ff_norm(x + self.dropout(_x))

        return x
        
class Encoder(nn.Module):
    def __init__(self, in_dim, h_dim, n_layers, num_heads, inter_dim, dropout_rate, device, seq_len):
        super(Encoder,self).__init__()
        self.token_embedding = nn.Embedding(in_dim, h_dim)
        self.positional_encoding = PositionalEncoding(h_dim, seq_len)
        self.layers = nn.ModuleList([EncoderLayer(h_dim, num_heads, inter_dim, dropout_rate, device) for i in range(n_layers)])
        self.dropout = nn.Dropout(dropout_rate)
        self.h_dim = h_dim
        self.device = device

    def forward(self,x, x_masking):
        x_token = self.token_embedding(x)
        scale = torch.sqrt(torch.FloatTensor([self.h_dim])).to(self.device)
        x_token = self.dropout(x_token * scale)
        out = self.positional_encoding(x_token).to(self.device)
        for layer in self.layers:
            out = layer(out, x_masking)
        return out

class DecoderLayer(nn.Module):
    def __init__(self, h_dim, num_heads, inter_dim, dropout_rate, device):
        super(DecoderLayer,self).__init__()
        self.norm = nn.LayerNorm(h_dim)
        self.encode_norm = nn.LayerNorm(h_dim)
        self.ff_norm = nn.LayerNorm(h_dim)
        self.attention_layer = MultiHeadAttentionLayer(h_dim, num_heads, dropout_rate, device)
        self.encode_attention_layer = MultiHeadAttentionLayer(h_dim, num_heads, dropout_rate, device)
        self.feedforward_layer = PositionWiseFeedForwardLayer(h_dim, inter_dim, dropout_rate)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, y, y_masking, x, x_masking):
        # 1. Masked Multi-Head Attention
        _y, _ = self.attention_layer(y, y, y, y_masking)
        
        # 2. Add & norm
        y = self.norm(y + self.dropout(_y))

        # 3. Encode Multi-Head Attention
        _y, attention = self.encode_attention_layer(y, x, x, x_masking)

        # 4. Add & norm
        y = self.encode_norm(y + self.dropout(_y))

        # 5. Positional Feed Forward
        _y = self.feedforward_layer(y)

        # 6. Add & norm
        y = self.ff_norm(y + self.dropout(_y))

        return y, attention


class Decoder(nn.Module):
    def __init__(self, out_dim, h_dim, n_layers, num_heads, inter_dim, dropout_rate, device, seq_len):
        super(Decoder,self).__init__()
        self.token_embedding = nn.Embedding(out_dim, h_dim)
        self.positional_encoding = PositionalEncoding(h_dim, seq_len)
        self.layers = nn.ModuleList([DecoderLayer(h_dim, num_heads, inter_dim, dropout_rate, device) for i in range(n_layers)])
        self.fc_out = nn.Linear(h_dim, out_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.h_dim = h_dim
        self.device = device

    def forward(self, y, y_masking, x, x_masking):
        y_token = self.token_embedding(y)
        scale = torch.sqrt(torch.FloatTensor([self.h_dim])).to(self.device)
        y_token = self.dropout(y_token * scale)
        out = self.positional_encoding(y_token).to(self.device)
        for layer in self.layers:
            out, attention = layer(out, y_masking, x, x_masking)

        out = self.fc_out(out)
        #out = F.softmax(out, dim= -1)
        return out, attention


class Transformer(nn.Module):
    def __init__(self, encoder, decoder, x_pad_idx, y_pad_idx, device):
        super(Transformer, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.x_pad_idx = x_pad_idx
        self.y_pad_idx = y_pad_idx
        self.device = device

    def set_x_masking(self, x):        
        x_masking = (x != self.x_pad_idx).unsqueeze(1).unsqueeze(2)

        return x_masking
    
    def set_y_masking(self, y):
        y_masking = (y != self.y_pad_idx).unsqueeze(1).unsqueeze(2)
        seq_len = y.shape[1]
        y_sub_masking = torch.tril(torch.ones((seq_len, seq_len), device = self.device)).bool()
        y_masking = y_masking & y_sub_masking

        return y_masking

    def forward(self,x,y):
        x_masking = self.set_x_masking(x)
        y_masking = self.set_y_masking(y)
        encoded_x = self.encoder(x, x_masking)
        out, attention = self.decoder(y, y_masking, encoded_x, x_masking)
        return out, attention

