#!/usr/bin/env python
# coding: utf-8

# # 数据处理模块

# ## 目录
# 
# * 数据集加载
# * 读取标签和数据
# * 创建word2id
# * 将数据转化成id

# In[1]:


from torch.utils import data
import os
import csv
import nltk
import numpy as np


# In[7]:


# 数据集加载
f = open("./data/AG/train.csv")
rows = csv.reader(f,delimiter=',', quotechar='"')
rows = list(rows)
rows[0:5]


# In[10]:


# 读取标签和数据
n_gram = 2 
lowercase = True
label = []
datas = []
for row in rows:
    label.append(int(row[0])-1)
    txt = " ".join(row[1:])
    if lowercase:
        txt = txt.lower()
    txt = nltk.word_tokenize(txt)   # 将句子转化成词
    new_txt=  []
    for i in range(0,len(txt)):
        for j in range(n_gram):   # 添加n-gram词
            if j<=i:
                new_txt.append(" ".join(txt[i-j:i+1]))
    datas.append(new_txt)
print (label[0:5])
print (datas[0:5])


# In[4]:


# 得到word2id
min_count = 3
word_freq = {}
for data in datas:   # 首先统计词频，后续通过词频过滤低频词
    for word in data:
        if word_freq.get(word)!=None:  
            word_freq[word]+=1
        else:
            word_freq[word] = 1
word2id = {"<pad>":0,"<unk>":1} 
for word in word_freq:   # 首先构建uni-gram词，因为不需要进行hashing trick
    if word_freq[word]<min_count or " " in word: # 小于min_count的单词进行过滤，而有空格在内说明是n-gram词对，之后处理
        continue
    word2id[word] = len(word2id)
uniwords_num = len(word2id) # unigram的个数
for word in word_freq:  # 构建2-gram以上的词，需要进行hashing trick
    if word_freq[word]<min_count or " " not in word: # 过滤小于min_count的n-gram词对，没有空格在内说明是unigram，已经处理过了
        continue
    word2id[word] = len(word2id) # 首先给n-gram的词对也分配一个id
word2id


# In[9]:


print (list(word2id.items())[-20:])
print (len(word2id))


# In[15]:





# In[6]:


# 将文本中的词都转化成id
max_length = 100
for i,data in enumerate(datas):
    for j,word in enumerate(data):
        if " " not in word: # 对于unigram的词
            datas[i][j] = word2id.get(word, 1) # 如果word不存在word2id字典中，则返回1，即<unk>的id
        else: # 对于n-gram的词对
            
            # 290000              % 100000                          + 30456               = 120456
            # n-gram一一对应的id  % 希望得到的n-gram的词向量的个数  + unigram的词的个数   = 这个n-gram词的词向量的id（可能有多个词共享同一个词向量id）
            datas[i][j] = word2id.get(word, 1) % 100000 + uniwords_num  # hash函数
    datas[i] = datas[i][0:max_length]+[0]*(max_length-len(datas[i])) # padding
datas[0:5]

