#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import torch.nn.functional as F


# In[4]:


class SkipGramModel(nn.Module):
    def __init__(self,embed_size,embed_dimension):
        super(SkipGramModel,self).__init__()
        self.embed_size = embed_size # 词表长度
        self.embed_dimension = embed_dimension # 每个词向量长度
        
        # nn.embedding中的参数sparse = True，输入参数是Index，如果sparse = False，则输入参数为one-hot编码
        self.w_embeddings = nn.Embedding(embed_size,embed_dimension,sparse=True) # 中心词embedding层
        self.v_embeddings = nn.Embedding(embed_size, embed_dimension, sparse=True) # 周围词embedding层
        self._init_emb()

    def _init_emb(self): # 初始化词向量矩阵权重
        initrange = 0.5 / self.embed_dimension
        self.w_embeddings.weight.data.uniform_(-initrange, initrange)
        self.v_embeddings.weight.data.uniform_(-0, 0)

    def forward(self, pos w, pos v, neg v):

        # 中间词的词向量，[mini_batch_size * emb_dimension]
        emb_w = self.w_embeddings(torch.Longtensor(pos_w))

        # 周围词的词向量，[mini_batch_size * emb_dimension]
        emb_v = self.v_embeddings(torch.Longtensor(pos_v))

        # 负采样的周围词词向量，[mini_batch_size * negative_sampling_number * emb_dimension]
        neg_emb_v =self.v_embeddings(torch.Longtensor(neg_v))

        # 正样本的score，中间词与正样本周围词词向量元素相乘
        # 为什么不是矩阵相乘？因为中间词词向量矩阵中每一行仅对应周围词词向量中的每一行，如果将周围词词向量转置后进行矩阵相乘，则会得到[mini_batch_size * mini_batch_size]大小的矩阵，其中代表的是每个中间词向量与每个周围词向量配对的概率，与输入的样本不同，因此进行元素间相乘，再在行方向上相加即可。
        score = torch.mul(emb_w, emb_v)
        score = torch.sum(score, dim = 1)
        score = F.logsigmoid(score)

        # 负样本的score，需要将中心词词向量增加一个维度并进行维度相乘，因为每个中心词对应negative_sampling_number个负样本周围词。
        # 这里采用的是torch.bmm的方法，即忽略第一个batch_size的维度，对后面两个维度进行矩阵相乘，这样一来，就是一个中心词词向量与其对应的负采样的多个周围词词向量进行矩阵相乘。
        neg_score = torch_bmm(neg_emb_v, emb_w.unsqueeze(2)) 
        neg_score = F.logsigmoid(-1 * neg_score)

        # 将正样本和负样本的score相加，要求正样本的score越大越好，负样本的score越小越好，因为neg_score已经 * -1，因此在loss中直接相加即可
        loss = -1 * (torch.sum(score) + torch.sum(neg_score))
        return loss

    def save_embedding(self, id2word, file_name): # 保存两个词向量
        embedding_1 = self.w_embeddings.weight.data.cpu().numpy() # 中心词词向量矩阵
        embedding_2 = self.v_embeddings.weight.data.cpu().numpy() # 周围词词向量矩阵
        embedding = (embedding_1+embedding_2)/2 # 相加/2，得到最终词向量矩阵
        fout = open(file_name, 'w')
        fout.write('%d %d\n' % (len(id2word), self.embed_dimension))
        for wid, w in id2word.items():
            e = embedding[wid]
            e = ' '.join(map(lambda x: str(x), e))
            fout.write('%s %s\n' % (w, e))


# In[6]:


model = SkipGramModel(100, 10)
id2word = dict()
for i in range(100):
    id2word[i] = str(i)
pos_w = [0, 0, 1, 1, 1]
pos_v = [1, 2, 0, 2, 3]
neg_v = [[23, 42, 32], [32, 24, 53], [32, 24, 53], [32, 24, 53], [32, 24, 53]]
model.forward(pos_w, pos_v, neg_v)


# In[9]:


model.w_embeddings.weight.data.cpu().numpy().shape


# In[ ]:




