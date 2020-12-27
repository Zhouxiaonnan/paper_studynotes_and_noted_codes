#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import torch.nn.functional as F


# In[4]:


class SkipGramModel(nn.Module):
    def __init__(self, emb_size, emb_dimension):
        super(SkipGramModel, self).__init__()
        self.emb_size = emb_size # 词表长度
        self.emb_dimension = emb_dimension # 词向量长度
        
        # 注意，因为Huffman树中内部节点的数量为词表长度-1，因此节点id数量有 2 * emb_sizee - 1
        self.w_embeddings = nn.Embedding(2*emb_size-1, emb_dimension, sparse=True) # 中间词
        self.v_embeddings = nn.Embedding(2*emb_size-1, emb_dimension, sparse=True) # 周围词
        self._init_emb()

    def _init_emb(self): # 初始化词向量矩阵
        initrange = 0.5 / self.emb_dimension
        self.w_embeddings.weight.data.uniform_(-initrange, initrange)
        self.v_embeddings.weight.data.uniform_(-0, 0)

    def forward(self, pos_w, pos_v,neg_w, neg_v):
        # 中心词正样本
        emb_w = self.w_embeddings(torch.LongTensor(pos_w)) 
        
        # 中心词负样本
        neg_emb_w = self.w_embeddings(torch.LongTensor(neg_w))
        
        # 周围词正样本
        emb_v = self.v_embeddings(torch.LongTensor(pos_v))
        
        # 周围词负样本
        neg_emb_v = self.v_embeddings(torch.LongTensor(neg_v))
        
        # 正样本score，这个score越大越好
        score = torch.mul(emb_w, emb_v).squeeze()
        score = torch.sum(score, dim=1)
        score = torch.clamp(score, max=10, min=-10)
        score = F.logsigmoid(score)
        
        # 负样本score，这个neg_score越小越好
        neg_score = torch.mul(neg_emb_w, neg_emb_v).squeeze()
        neg_score = torch.sum(neg_score, dim=1)
        neg_score = torch.clamp(neg_score, max=10, min=-10)
        neg_score = F.logsigmoid(-neg_score)
        
        # L = log sigmoid (Xw.T * θv) + [log sigmoid (-Xw.T * θv)]
        loss = -1 * (torch.sum(score) + torch.sum(neg_score))
        return loss
    
    # 注意，这里保存的embedding有2 * emb_size - 1个词向量，前面的词向量为单词的词向量，后面的词向量包括根节点和内部节点的词向量
    def save_embedding(self, id2word, file_name): 
        embedding = self.w_embeddings.weight.data.cpu().numpy()
        fout = open(file_name, 'w')
        fout.write('%d %d\n' % (len(id2word), self.emb_dimension))
        for wid, w in id2word.items():
            e = embedding[wid]
            e = ' '.join(map(lambda x: str(x), e))
            fout.write('%s %s\n' % (w, e))


# In[5]:


model = SkipGramModel(100, 10)
id2word = dict()
for i in range(100):
    id2word[i] = str(i)
pos_w = [0, 0, 1, 1, 1]
pos_v = [1, 2, 0, 2, 3]
neg_w = [0, 0, 1, 1, 1]
neg_v = [54,55, 61, 71, 82]
model.forward(pos_w, pos_v, neg_w,neg_v)


# In[8]:


model.v_embeddings.weight.data.cpu().numpy().shape


# In[ ]:




