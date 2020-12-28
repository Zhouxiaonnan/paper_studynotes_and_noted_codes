#!/usr/bin/env python
# coding: utf-8

# # 数据处理模块

# ## 目录
# 
# * 数据集加载
# * 构建word2id并去除低频词
# * 构建共现矩阵
# * 生成训练集
# * 保存结果

# In[3]:


from torch.utils import data
import os
import numpy as np
import pickle


# In[2]:


min_count = 50


# In[3]:


# 数据集加载
data = open("./data/text8.txt").read()
data = data.split()
# 构建word2id并去除低频词
word2freq = {}
for word in data:
    if word2freq.get(word)!=None:
        word2freq[word] += 1
    else:
        word2freq[word] = 1
word2id = {}
for word in word2freq:
    if word2freq[word]<min_count:
        continue
    else:
        if word2id.get(word)==None:
            word2id[word]=len(word2id)
print (len(word2id))
word2id


# In[4]:


# 构建共现矩阵
vocab_size = len(word2id)
comat = np.zeros((vocab_size,vocab_size))
print(comat.shape)


# In[5]:


window_size = 2


# In[6]:


# 共现矩阵
for i in range(len(data)):
    if i%1000000==0:
        print (i,len(data))
    if word2id.get(data[i])==None:
        continue
    w_index = word2id[data[i]]
    for j in range(max(0,i-window_size),min(len(data),i+window_size+1)):
        if word2id.get(data[j]) == None or i==j:
            continue
        u_index = word2id[data[j]]
        comat[w_index][u_index]+=1 # 如果在窗口长度内共同出现则+1
comat


# In[7]:


coocs = np.transpose(np.nonzero(comat)) # 得到不为0的元素的坐标并转置
coocs


# In[4]:


# np.nonzero(np.array([[1,0,0], [0,1,1]]))


# In[5]:


# np.transpose(np.nonzero(np.array([[1,0,0], [0,1,1]])))


# In[9]:


# 生成训练集
labels = []
for i in range(len(coocs)):
    if i%1000000==0:
        print (i,len(coocs))
    labels.append(comat[coocs[i][0]][coocs[i][1]]) # 得到所有不为0的元素的值，与位置一一对应
labels = np.array(labels)
print (labels.shape)


# In[10]:


labels


# In[12]:


# 保存结果
np.save("./data/data.npy",coocs)
np.save("./data/label.npy",labels)
pickle.dump(word2id,open("./data/word2id","wb"))


# In[ ]:




