#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import numpy as np


# In[11]:


class Fasttext(nn.Module):
    def __init__(self,vocab_size,embedding_size,max_length,label_num):
        super(Fasttext,self).__init__()
        
        # vocab_size 词表大小
        # embedding_size 词向量长度
        self.embedding =nn.Embedding(vocab_size,embedding_size)  # 嵌入层
        
        # max_length 句子的最大长度
        self.avg_pool = nn.AvgPool1d(kernel_size=max_length,stride=1) # 平均层，即将词向量进行平均
        self.fc = nn.Linear(embedding_size, label_num) # 全连接层
        
    def forward(self, x):
        x = x.long()
        out = self.embedding(x) # batch_size * sentence_max_length * embedding_size
        out = out.transpose(1, 2).contiguous() # batch_size * embedding_size * sentence_max_length
        out = self.avg_pool(out).squeeze() # batch_size * embedding_size
        out = self.fc(out) # batch_size * label_num
        return out


# In[12]:


fasttext = Fasttext(vocab_size=1000,embedding_size=10,max_length=100,label_num=4)
test = torch.zeros([64,100]).long()
out = fasttext(test)


# In[13]:


out.shape


# In[14]:


from torchsummary import summary
# lenet-5 


# In[15]:


summary(fasttext, input_size=(100,))


# In[ ]:




