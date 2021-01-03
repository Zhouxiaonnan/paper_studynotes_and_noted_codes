#!/usr/bin/env python
# coding: utf-8

# # 数据处理模块

# ## 目录
# 
# * 数据集加载
# * 根据长度排序
# * 创建word2id
# * 分句
# * 将数据转化成id

# In[4]:


from torch.utils import data
import os
import nltk
import numpy as np
import pickle
from collections import Counter


# In[6]:


# 数据集加载
datas = open("./data/imdb/imdb-test.txt.ss",encoding="utf-8").read().splitlines()
datas = [data.split("		")[-1].split()+[data.split("		")[2]] for data in datas] # 将data与lable放在一起，排序时不会打乱
datas[0:5]


# In[7]:


# 根据长度排序，这样每个batch中句子可以进行较少的padding，计算时间可以下降
datas = sorted(datas,key = lambda x:len(x),reverse=True) # 为什么从最长的开始训练？因为如果从最短的开始，那么训练到最长的时候可能显存不足
labels  = [int(data[-1])-1 for data in datas]
datas = [data[0:-1] for data in datas]
print(labels[0:5])
print (datas[-5:])


# In[9]:


# word2id
min_count = 5
word_freq = {}
for data in datas:
    for word in data:
        word_freq[word] = word_freq.get(word,0)+1
word2id = {"<pad>":0,"<unk>":1}
for word in word_freq:
    if word_freq[word]<min_count:
        continue
    else:
        word2id[word] = len(word2id)
word2id


# In[10]:


# 分句
for i,data in enumerate(datas):
    datas[i] = " ".join(data).split("<sssss>")
    for j,sentence in enumerate(datas[i]):
        datas[i][j] = sentence.split()
datas[0]


# In[11]:


# 将数据转化为id
max_sentence_length = 100 # 句子必须一样的长度
batch_size = 64 # 每个batch size，每个文档的句子一样多
for i,document in enumerate(datas):
    if i%10000==0:
        print (i,len(datas))
    for j,sentence in enumerate(document):
        for k,word in enumerate(sentence):
            datas[i][j][k] = word2id.get(word,word2id["<unk>"])
        datas[i][j] = datas[i][j][0:max_sentence_length] + \ # 截断
                      [word2id["<pad>"]]*(max_sentence_length-len(datas[i][j])) # pad
for i in range(0,len(datas),batch_size):
    max_data_length = max([len(x) for x in datas[i:i+batch_size]])
    for j in range(i,min(i+batch_size,len(datas))):
        datas[j] = datas[j] + [[word2id["<pad>"]]*max_sentence_length]*(max_data_length-len(datas[j]))
datas[0]


# In[12]:


import numpy as np
for i in range(0,len(datas),64):
    batch_datas = np.array(datas[i:i+64])
    batch_labels = np.array(labels[i:i+64])
    print (batch_datas.shape)
    print (batch_labels.shape)
    


# In[ ]:




