#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn


# In[2]:


class glove_model(nn.Module):
    def __init__(self,vocab_size,embed_size,x_max,alpha):
        super(glove_model, self).__init__()
        self.vocab_size = vocab_size  # 词表长度
        self.embed_size = embed_size  # 词向量维度
        self.x_max = x_max # 词对最大出现次数，用于计算词对的权重
        self.alpha = alpha # 超参数，用于计算词对的权重
        self.w_embed = nn.Embedding(self.vocab_size,self.embed_size).type(torch.float64) # 中心词向量

        self.w_bias = nn.Embedding(self.vocab_size,1).type(torch.float64) # 中心词bias

        self.v_embed = nn.Embedding(self.vocab_size, self.embed_size).type(torch.float64) # 周围词向量

        self.v_bias = nn.Embedding(self.vocab_size, 1).type(torch.float64) # 周围词bias
        
    def forward(self, w_data,v_data,labels):
        w_data_embed = self.w_embed(w_data) # 中心词 bs * embed_size
        w_data_bias = self.w_bias(w_data) # 中心词bias bs * 1
        v_data_embed = self.v_embed(v_data) # 周围词 bs * embed_size
        v_data_bias = self.v_bias(v_data) # 周围词bias bs * 1
        weights = torch.pow(labels / self.x_max, self.alpha) # 权重生成，词对出现超过100次，则权重直接设置为1
        weights[weights>1]=1 # 权重不能太大，防止常出现的词的影响（are, is, the, a等）
        loss = torch.mean(weights * # 权重
                          torch.pow( # 平方
                                  torch.sum(
                                          w_data_embed * v_data_embed, 1 # 矩阵相乘
                                          ) + 
                                  w_data_bias + 、# 中心词bias
                                  v_data_bias -   # 周围词bias
                                 torch.log(labels),  # log(词对出现次数)
                                    2)) # 计算loss
        return loss
    
    def save_embedding(self, word2id, file_name):
        embedding_1 = self.w_embed.weight.data.cpu().numpy() # 中心词词向量
        embedding_2 = self.v_embed.weight.data.cpu().numpy() # 周围词词向量
        embedding = (embedding_1 + embedding_2) / 2 # 两个词向量的平均值
        fout = open(file_name, 'w')
        fout.write('%d %d\n' % (len(word2id), self.embed_size))
        for w, wid in word2id.items():
            e = embedding[wid]
            e = ' '.join(map(lambda x: str(x), e))
            fout.write('%s %s\n' % (w, e))


# In[3]:


model = glove_model(100, 100,100,0.75)
word2id = dict()
for i in range(100):
    word2id[str(i)] = i
w_data = torch.Tensor([0, 0, 1, 1, 1]).long() # 中心词
v_data =  torch.Tensor([1, 2, 0, 2, 3]).long() # 周围词
labels = torch.Tensor([1,2,3,4,5]) # 词对出现的次数
model.forward(w_data, v_data, labels)


# In[4]:


embedding_1 = model.w_embed.weight.data.cpu().numpy()


# In[5]:


embedding_1.shape


# In[ ]:




