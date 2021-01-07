#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# %load E:\论文\transformer\annotated-Transformer-notebook\model\model.py


import torch.nn as nn
import torch.nn.functional as F
import copy
from .attention import MultiHeadedAttention
from .otherlayer import Embeddings,PositionalEncoding,PositionwiseFeedForward
from .encoderblock import Encoder,EncoderLayer
from .decoderblock import Decoder,DecoderLayer
import seaborn
seaborn.set_context(context="talk")

"""
   定义encoder-decoder框架
"""
class EncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many
    other models.
    """

    # super(EncoderDecoder, self).__init__() 等价于 nn.Module.__init__()
    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed # encoder embedding + positional encoding
        self.tgt_embed = tgt_embed # decoder embedding + positional encoding
        self.generator = generator # 最后的线性层 + softmax

    # forward函数调用自身encode方法实现encoder，然后调用decode方式实现decoder
    def forward(self, src, tgt, src_mask, tgt_mask):
        # src的shape=tgt的shape：[batch_size,max_length]
        "Take in and process masked src and target sequences."
        return self.decode(self.encode(src, src_mask),  # 先经过 encoder
                           src_mask, tgt, tgt_mask) # 再经过 decoder
    
    # encode embedding + positional encoding
    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)

    # decode embedding + positional encoding
    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)

# 最后的线性层 + softmax 层
class Generator(nn.Module):
    "Define standard linear + softmax generation step."
    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        #nn.Linear是一个线性层，x*W
        return F.log_softmax(self.proj(x), dim=-1)


def make_model(src_vocab,   # 原语言词表大小
               tgt_vocab,   # 目标语言词表大小
               N=6,         # encoder 和 decoder 的数量
               d_model=512, # embedding_size
               d_ff=2048,   # 线性层维度变换
               h=8,         # multihead attention 的头数量
               dropout=0.1):
    c = copy.deepcopy # 复制
    
    # 多头注意力机制
    attn = MultiHeadedAttention(h, # 输入希望进行多少个 Head
                                d_model) # 以及 embedding_size
    
    ff = PositionwiseFeedForward(d_model, d_ff, dropout) # 前馈神经网络
    position = PositionalEncoding(d_model, dropout) # 位置编码
    model = EncoderDecoder(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N), # encoder
        Decoder(DecoderLayer(d_model, c(attn), c(attn), # decoder
                             c(ff), dropout), N),
        nn.Sequential(Embeddings(d_model, src_vocab), c(position)), # encoder embedding + encoder position encoding
        nn.Sequential(Embeddings(d_model, tgt_vocab), c(position)), # decoder embedding + decoder position encoding 
        Generator(d_model, tgt_vocab)) # 最后的线性层 + softmax

    # This was important from their code.
    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform(p)
    return model


if __name__ == '__main__':
    make_model(10, 10, 2)


# In[ ]:


# %load E:\论文\transformer\annotated-Transformer-notebook\model\encoderblock.py


import torch.nn as nn
from .subutils import clones,LayerNorm,SublayerConnection


# Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N)
class Encoder(nn.Module):
    "Core encoder is a stack of N layers"
    
    # layer：encoderlayer
    # N：N 个 encoderlayer
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        # [batch,max,embe]
        "Pass the input (and mask) through each layer in turn."
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

#传入参数
#EncoderLayer(d_model, c(attn), c(ff), dropout), N
class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn # self attention 层
        self.feed_forward = feed_forward # 前馈层
        self.sublayer = clones(SublayerConnection(size, dropout), 2) # 在一个 encoder 中进行两次残差相加
        self.size = size

    def forward(self, x, mask):
        "Follow Figure 1 (left) for connections."
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask)) # 经过 multihead attention 后的残差相加
        return self.sublayer[1](x, self.feed_forward) # 经过前馈后的残差相加


# In[ ]:


# %load E:\论文\transformer\annotated-Transformer-notebook\model\decoderblock.py

import torch.nn as nn
from .subutils import clones,LayerNorm,SublayerConnection

# 传入参数
# Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N)
class Decoder(nn.Module):
    "Generic N layer decoder with masking."

    # layer：decoderlayer
    # N：进行 N 次 decoderlayer
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)


# 传入参数
# DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N
class DecoderLayer(nn.Module):
    "Decoder is made of self-attn, src-attn, and feed forward (defined below)"

    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn # self attention
        self.src_attn = src_attn # encoder - decoder attention
        self.feed_forward = feed_forward # 前馈
        self.sublayer = clones(SublayerConnection(size, dropout), 3) # 一个 decoderlayer 中进行了三次残差

    # memory就是encoder完成之后的结果
    def forward(self, x, memory, src_mask, tgt_mask):
        "Follow Figure 1 (right) for connections."
        m = memory # encoder 输出

        # self-attetion q=k=v,输入是decoder的embedding+positonal
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask)) # self-attention + 残差

        # soft-attention q!=k=v x是deocder的embedding，m是encoder的输出
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask)) # encoder-decoder attention + 残差
        return self.sublayer[2](x, self.feed_forward) # 前馈 + 残差


# In[ ]:


# %load E:\论文\transformer\annotated-Transformer-notebook\model\attention.py



import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time
from .subutils import clones

"""
    Scaled Dot product attention
"""
def attention(query, key, value, mask=None, dropout=None):
    # shape:query=key=value---->[batch_size,8,max_length,64]

    d_k = query.size(-1)

    # k的纬度交换后为：[batch_size,8,64,max_length]
    # scores的纬度为:[batch_size,8,max_length,max_length]
    scores = torch.matmul(query, key.transpose(-2, -1)) \ # Q * K^T
             math.sqrt((d_k), #, Q, *, K^T, /, sqrt(dk))

    # padding mask
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)

    p_attn = F.softmax(scores, dim = -1) # softmax(Q * K^T / sqrt(dk))
    if dropout is not None:
        p_attn = dropout(p_attn)
        
    # softmax(Q * K^T / sqrt(dk)) * V
    return torch.matmul(p_attn, value), p_attn


"""
    MultiHeadAttention
"""
class MultiHeadedAttention(nn.Module):
    def __init__(self, 
                 h, # multihead attention 的头数量
                 d_model, # embedding_size
                 dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h # 分头后的维度
        self.h = h # 头的个数
        self.linears = clones(nn.Linear(d_model, d_model), 4) # 4个 d_model * d_model 的层
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        # 纬度
        # shape:query=key=value--->:[batch_size,max_legnth,embedding_dim=512]

        if mask is not None:
            # Same mask applied to all h heads.
            # 因为 mask 传进来是3个维度，单只对应一个 Head，因此需要升到4维
            mask = mask.unsqueeze(1)
        nbatches = query.size(0) # batch_size

        # 第一步：将q,k,v分别与Wq，Wk，Wv矩阵进行相乘
        # shape:Wq=Wk=Wv----->[512,512]
        # 第二步：将获得的Q、K、V在第三个纬度上进行切分
        # shape:[batch_size,max_length,8,64]
        # 第三部：填充到第一个纬度
        # shape:[batch_size,8,max_length,64]
        query, key, value = \
        # 将 query, key, value 分别通过一个线性层（相当于添加了权重和偏置）并切分维度，得到新的 query, key, value
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2) 
             for l, x in zip(self.linears, (query, key, value))] # 取出 self.linears 四个中的三个

        # 进入到attention之后纬度不变，shape:[batch_size,8,max_length,64]
        # 因为 attention 是对最后两个维度进行计算，因此一次输入，相当于同时进行了8个attention，即 Multihead attention
        # scaled dot product attention
        # x：attention 的结果
        # self.attn：在 attention 中 softmax 后的概率
        x, self.attn = attention(query, key, value, mask=mask,
                                 dropout=self.dropout)

        # 将纬度进行还原，相当于将8个 Head 的结果进行 concat
        # 交换纬度：[batch_size,max_length,8,64]
        # 纬度还原：[batch_size,max_length,512]
        x = x.transpose(1, 2).contiguous() \
            .view(nbatches, -1, self.h * self.d_k)

        # 最后与WO大矩阵相乘 shape:[512,512]
        return self.linears[-1](x)


# In[ ]:


# %load E:\论文\transformer\annotated-Transformer-notebook\model\otherlayer.py



import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.autograd import Variable

"""
    主要包括feedforwad-network，positonal-encoding，embedding
"""

# FFN
class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."
    # d_model: embedding_size
    # d_ff: 线性层维度变换
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff) # 线性层1，[512,2048]
        self.w_2 = nn.Linear(d_ff, d_model) # 线性层2，[2048,512]
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x)))) # 线性层1 + relu + 线性层2

# embedding
class Embeddings(nn.Module):
    
    # d_model: embedding_size
    # vocab: 词表大小
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model) # 论文中提到再乘以 sqrt(d_model)


class PositionalEncoding(nn.Module):
    "Implement the PE function."
    
    # d_model: embedding_size，位置编码长度需要与embedding一致
    # max_len: max_sentence_length
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model) # [max_sentence_length * embedding_size]
        position = torch.arange(0., max_len).unsqueeze(1) # 位置 [max_sentence_length, 1]
        
        # 开始编码
        div_term = torch.exp(torch.arange(0., d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe) # 序列化
        
        # 序列化:
        # 将对象的状态信息转换为可以存储或传输的形式的过程。
        # 在序列化期间，对象将其当前状态写入到临时或持久性存储区。
        # 以后，可以通过从存储区中读取或反序列化对象的状态，重新创建该对象。
    
    # x：embedding
    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], # embedding + positional encoding
                         requires_grad=False # 不回传梯度，即不进行训练
                        )
        return self.dropout(x)


# In[ ]:


# %load E:\论文\transformer\annotated-Transformer-notebook\model\subutils.py

import torch
import torch.nn as nn
import copy
import numpy as np
def clones(module, N):
    "Produce N identical layers."
    # copy.deepcopy硬拷贝函数
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

# 层标准化，对每一个样本的最后一层做标准化
# 此处即对 embedding_size 做标准化
class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        # features=layer.size=512
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps # 防止分母为0

    def forward(self, x):
        mean = x.mean(-1, keepdim=True) # mean
        std = x.std(-1, keepdim=True) # std
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2 # 标准化

# 残差
class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size) # 标准化
        self.dropout = nn.Dropout(dropout)

    # sublayer：进行残差相加和标准化层的下面一层（multihead attention 或 前馈层）
    # x：未经过 sublayer 的数据
    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))



# In[ ]:




