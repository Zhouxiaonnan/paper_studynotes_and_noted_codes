#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import torch.nn as nn
import numpy as np
from torch.nn import functional as F
from torch.autograd import Variable


# In[ ]:


class HAN_Model(nn.Module):
    def __init__(self,vocab_size,embedding_size,gru_size,class_num,is_pretrain=False,weights=None):
        super(HAN_Model, self).__init__()
        if is_pretrain:
            self.embedding = nn.Embedding.from_pretrained(weights, freeze=False)
        else:
            self.embedding = nn.Embedding(vocab_size, embedding_size) # vocab_size 词表大小， embedding_size 词向量长度
            
        # word 部分
        self.word_gru = nn.GRU(input_size=embedding_size,hidden_size=gru_size,num_layers=1,
                               bidirectional=True, # 双向GRU
                               batch_first=True)
        self.word_context = nn.Parameter(torch.Tensor(2 * gru_size, 1),requires_grad = True)
        self.word_dense = nn.Linear(2 * gru_size, 2 * gru_size)
    
        # sentence 部分
        self.sentence_gru = nn.GRU(input_size = 2 * gru_size,hidden_size=gru_size,num_layers=1,
                               bidirectional=True, # 双向GRU
                               batch_first=True)
        self.sentence_context = nn.Parameter(torch.Tensor(2 * gru_size, 1),requires_grad=True)
        self.sentence_dense = nn.Linear(2 * gru_size,2 * gru_size)
        self.fc = nn.Linear(2 * gru_size,class_num)
        
    def forward(self, x,gpu=False):
        sentence_num = x.shape[1] # 句子数量
        sentence_length = x.shape[2] # 句子长度
        
        '''word 部分'''
        # x: bs * sentence_num * sentence_length -> (bs * sentence_num) * sentence_length
        x = x.view([-1, sentence_length])
        
        # embedding
        # (bs * sentence_num) * sentence_length * embedding_size
        # 三维，最后一维是词向量
        x_embedding = self.embedding(x) 
        
        # BiGRU
        # word_outputs.shape: (bs * sentence_num) * sentence_length * (2 * gru_size)
        # 因为是双向 GRU，最后得到的 hidden state 是正向和反向 hidden state 的 concat
        word_outputs, word_hidden = self.word_gru(x_embedding) 
        
        # 全连接层 + tanh
        # word_outputs_attention.shape: (bs * sentence_num) * sentence_length * (2 * gru_size)
        word_outputs_attention = torch.tanh(self.word_dense(word_outputs)) 
        
        # 求内积
        # (bs * sentence_num) * sentence_length * 1
        weights = torch.matmul(word_outputs_attention, self.word_context)
        
        # softmax 得到每个 hidden state 的权重
        # (bs * sentence_num) * sentence_length * 1
        weights = F.softmax(weights,dim=1) 
        
        # 根据句子的 padding 分配权重（如果是 pad 的位置，则权重分配为0，其他真实存在单词的位置，分配权重）
        # (bs * sentence_num) * sentence_length * 1
        x = x.unsqueeze(2) 
        
        # torch.full_like，根据矩阵x的形状，生成一个全0矩阵
        # torch.where，判断x中不为0的位置，如果不是0，则返回weights中的权重，是0，则返回第三个全为0的矩阵中对应位置的值（即0）
        # 于是最终得到一个在句子有真实单词的位置有权重，在padding的位置权重为0的矩阵
        # (bs * sentence_num) * sentence_length * 1
        if gpu:
            weights = torch.where(x!=0,weights,torch.full_like(x,0,dtype=torch.float).cuda())
        else:
            weights = torch.where(x != 0, weights, torch.full_like(x, 0, dtype=torch.float)) 
        
        # 经过上面的处理，权重的值之和不为1了，所以需要进行一次重新计算，即每个权重/权重之和，使所有权重的和为1
        # +1e-4 是为了防止分母为0
        # (bs * sentence_num) * sentence_length * 1
        weights = weights/(torch.sum(weights,dim=1).unsqueeze(1)+1e-4) 
        
        '''sentence 部分'''
        # 首先将word部分得到的weights和每个词的hidden state进行加权求和
        # 转换矩阵形状，得到一个三维的矩阵
        # 因为word的 hidden state 的第三个维度的长度是 2 * gru_size，所以 sentence vector 也是
        # bs * sentence_num * (2 * gru_size)
        sentence_vector = torch.sum(word_outputs * weights,dim=1).view([-1,sentence_num,word_outputs.shape[-1]]) 
        
        # BiGRU
        # sentence_outputs.shape: bs * sentence_num * (2 * gru_size)
        sentence_outputs, sentence_hidden = self.sentence_gru(sentence_vector)
        
        # 全连接层 + tanh
        # attention_sentence_outputs.shape: bs * sentence_num * (2 * gru_size)
        attention_sentence_outputs = torch.tanh(self.sentence_dense(sentence_outputs)) 
        
        # 求内积
        # bs * sentence_num * 1
        weights = torch.matmul(attention_sentence_outputs,self.sentence_context) 
        
        # softmax
        # bs * sentence_num * 1
        weights = F.softmax(weights,dim=1)
        
        # 因为有些句子是padding的，即是一个不真实存在的句子，因此权重应该为0
        # bs * sentence_num * sentence_length
        x = x.view(-1, sentence_num, x.shape[1])
        
        # 这里求和是为了查看哪些句子是padding的句子，因为不是padding的句子，其第三个维度是单词的id，相加肯定不为0
        # 如果是padding的句子，内部所有位置都是0，因此相加肯定为0
        # bs * sentence_num * 1
        x = torch.sum(x, dim=2).unsqueeze(2)
        
        # 根据padding，得到不是padding句子的权重
        #  bs * sentence_num * 1
        if gpu:
            weights = torch.where(x!=0,weights,torch.full_like(x,0,dtype=torch.float).cuda())
        else:
            weights = torch.where(x != 0, weights, torch.full_like(x, 0, dtype=torch.float))
        
        # 重新计算句子的权重
        #  bs * sentence_num * 1
        weights = weights / (torch.sum(weights,dim=1).unsqueeze(1)+1e-4)
        
        # 句子权重与句子的hidden state加权求和
        # bs * (2 * gru_size)
        document_vector = torch.sum(sentence_outputs*weights,dim=1)
        
        # 全连接层
        # bs * class_num
        output = self.fc(document_vector) 
        return output


# In[ ]:


han_model = HAN_Model(vocab_size=30000,embedding_size=200,gru_size=50,class_num=4)
x = torch.Tensor(np.zeros([64,50,100])).long()
x[0][0][0:10] = 1
output = han_model(x)
print (output.shape)


# In[ ]:




