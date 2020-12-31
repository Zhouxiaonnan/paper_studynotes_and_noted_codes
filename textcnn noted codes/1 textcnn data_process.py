#!/usr/bin/env python
# coding: utf-8

# # 数据处理模块

# ## 目录
# 
# * 词向量导入
# * 数据集加载
# * 构建word2id并pad成相同长度
# * 求词向量均值和方差
# * 生成词向量
# * 生成训练集、验证集和测试集

# In[1]:


from torch.utils import data
import os
import random
import numpy as np
from gensim.test.utils import datapath, get_tmpfile
from gensim.models import KeyedVectors


# In[2]:


# 词向量导入
wvmodel = KeyedVectors.load_word2vec_format("../GoogleNews-vectors-negative300.bin.gz",binary=True)
wvmodel.get_vector("good")


# In[3]:


# 数据集加载
pos_samples = open("./data/MR/rt-polarity.pos",errors="ignore").readlines()
neg_samples = open("./data/MR/rt-polarity.neg",errors="ignore").readlines()
datas = pos_samples+neg_samples
datas = [data.split() for data in datas]
labels = [1]*len(pos_samples)+[0]*len(neg_samples)
print (len(datas),len(labels))


# In[4]:


pos_samples[:5]


# In[5]:


neg_samples[:5]


# In[7]:


np.array(datas[:5])


# In[8]:


# 构建word2id并pad成相同长度
max_sample_length = max([len(sample) for sample in datas])
word2id = {"<pad>":0}
for i,data in enumerate(datas):
    for j,word in enumerate(data):
        if word2id.get(word)==None:
            word2id[word] = len(word2id)
        datas[i][j] = word2id[word]
    datas[i] = datas[i]+[0]*(max_sample_length-len(datas[i])) #将所有句子pad成max_sample_length的长度
    #datas[i] = datas[i][0:max_sample_length]+[0]*(max_sample_length-len(datas[i]))  #包含截断的写法


# In[9]:


max_sample_length


# In[10]:


datas[0]


# In[11]:


# 求词向量均值和方差，用于未出现单词的词向量的初始化
tmp = []
for word, index in word2id.items():
    try:
        tmp.append(wvmodel.get_vector(word))
    except:
        pass
mean = np.mean(np.array(tmp)) # 词向量均值
std = np.std(np.array(tmp)) # 词向量方差
print (mean,std)


# In[12]:


# 生成词向量
vocab_size = len(word2id)
embed_size = 300
#embedding_weights = np.random.normal(-0.0016728516,0.17756976,[vocab_size,embed_size])
embedding_weights = np.random.normal(mean,std,[vocab_size,embed_size]) # 如果在word2vec词向量中找不到的则使用mean和std进行初始化
for word, index in word2id.items(): # 如果可以找到则使用word2vec的词向量
    try:
        embedding_weights[index, :] = wvmodel.get_vector(word)
    except:
        pass


# In[13]:


embedding_weights.shape


# In[14]:


# 打乱数据集
c = list(zip(datas,labels))
random.seed(1)
random.shuffle(c)
datas[:],labels[:] = zip(*c)


# In[15]:


datas[0]


# In[16]:


# 生成训练集、验证集和测试集
k = 0
# ；k=3 0,1,2+4-9


# In[17]:


train_datas = datas[:int(k * len(datas) / 10)] + datas[int((k + 1) * len(datas) / 10):]
train_labels = labels[:int(k * len(datas) / 10)] + labels[int((k + 1) * len(labels) / 10):]


# In[18]:


valid_datas = np.array(train_datas[int(0.9 * len(train_datas)):])
valid_labels = np.array(train_labels[int(0.9 * len(train_labels)):])


# In[19]:


print (valid_datas.shape,valid_labels.shape)


# In[20]:


train_datas = np.array(train_datas[0:int(0.9*len(train_datas))])
train_labels = np.array(train_labels[0:int(0.9*len(train_labels))])


# In[21]:


print (train_datas.shape,train_labels.shape)


# In[22]:


test_datas = np.array(datas[int(k * len(datas) / 10):int((k + 1) * len(datas) / 10)])
test_labels = np.array(labels[int(k * len(datas) / 10):int((k + 1) * len(datas) / 10)])


# In[23]:


print (test_datas.shape,test_labels.shape)


# In[ ]:




