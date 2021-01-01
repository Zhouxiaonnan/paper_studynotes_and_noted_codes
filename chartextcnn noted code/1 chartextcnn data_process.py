#!/usr/bin/env python
# coding: utf-8

# # 数据处理模块

# ## 目录
# 
# * 数据集加载
# * 读取标签和数据
# * 读取所有的字符
# * 将句子ont-hot表示

# In[1]:


import os
import torch
import json
import csv


# In[11]:


# 数据集加载
f = open("./data/AG/train.csv")
datas = csv.reader(f,delimiter=',', quotechar='"')
datas = list(datas)
datas[0:5]


# In[13]:


# 读取标签和数据
label = []
data = []
lowercase = True
for row in datas:
    label.append(int(row[0]-1))
    txt = " ".join(row[1:])
    if lowercase:
        txt = txt.lower()
    data.append(txt)
print (label[0:5])
print (data[0:5])


# In[14]:


# 读取所有的字符
with open("./data/alphabet.json") as f:
    alphabet = "".join(json.load(f))
alphabet


# In[15]:


def char2Index(char):
    return  alphabet.find(char) # 如果存在则返回1，不存在则返回-1


# In[17]:


char2Index("b")


# In[18]:


char2Index("我")


# In[21]:


# 将句子ont-hot表示
l0 = 1014
def oneHotEncode(idx):
    X = torch.zeros(l0,len(alphabet)) # one-hot 向量长度为 len(alphabet)，对应的字符为1，其他为0
    for index_char, char in enumerate(data[idx]):
        if char2Index(char)!=-1:
            X[index_char][char2Index(char)] = 1.0
    return X


# In[23]:


oneHotEncode(0)[0:5]


# In[ ]:


# torch 1d卷积和1d池化：batch_size*feature*length

