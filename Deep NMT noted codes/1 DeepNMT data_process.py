#!/usr/bin/env python
# coding: utf-8

# # 数据处理模块

# ## 目录
# 
# * 数据集加载
# * 读取双语语料
# * 创建word2id
# * 将数据转化成id

# In[4]:


from torch.utils import data
import os
import nltk
import numpy as np
import pickle
from collections import Counter


# In[5]:


# 数据集加载
raw_source_data = open("./data/iwslt14/train.tags.de-en.de",encoding="utf-8").readlines() # 原语言
raw_target_data = open("./data/iwslt14/train.tags.de-en.en",encoding="utf-8").readlines() # 翻译成的语言
raw_source_data = [x[0:-1] for x in raw_source_data]
raw_target_data = [x[0:-1] for x in raw_target_data]
print (len(raw_target_data))
print (len(raw_source_data))
print (raw_source_data[0:5])
print (raw_target_data[0:5])


# In[6]:


source_data = []
target_data = []
for i in range(len(raw_source_data)):
    if raw_target_data[i]!="" and raw_source_data[i]!="" and raw_source_data[i][0]!="<" and raw_target_data[i][0]!="<": # 去除网址和空
        source_sentence = nltk.word_tokenize(raw_source_data[i],language="german") # nltk.word_tokenize 分词工具
        target_sentence = nltk.word_tokenize(raw_target_data[i],language="english")
        if len(source_sentence)<=100 and len(target_sentence)<=100: # max_sentence_length
            source_data.append(source_sentence)
            target_data.append(target_sentence)
print (source_data[0:5])
print (target_data[0:5])


# In[8]:


# 源语言word2id
words = []
for sentence in source_data:
    for word in sentence:
        words.append(word)
word_freq = dict(Counter(words).most_common(30000-4))
source_word2id = {"<pad>":0,"<unk>":1,"<start>":2,"<end>":3}
for word in word_freq:
    source_word2id[word] = len(source_word2id)
source_word2id


# In[9]:


# 目标语言word2id
words = []
for sentence in target_data:
    for word in sentence:
        words.append(word)
word_freq = dict(Counter(words).most_common(30000-4))
target_word2id = {"<pad>":0,"<unk>":1,"<start>":2,"<end>":3}
for word in word_freq:
    target_word2id[word] = len(target_word2id)
target_word2id


# In[10]:


# 源语言数据转id
for i, sentence in enumerate(source_data):
    for j, word in enumerate(sentence):
        source_data[i][j] = source_word2id.get(word,1) # 转ID，如果不存在这个word，则为1，即<unk>
    source_data[i] = source_data[i][0:100] +[0]*(100-len(source_data[i])) # padding
    source_data[i].reverse()
source_data[0:5]


# In[11]:


# 目标语言数据转id
for i, sentence in enumerate(target_data):
    for j, word in enumerate(sentence):
        target_data[i][j] = target_word2id.get(word,1)
    target_data[i] = target_data[i][0:99]+ [3] + [0] * (99 - len(target_data[i]))
target_data[0:5]


# In[12]:


# 训练错位输入
# 在训练的过程中，decoder部分每个cell需要根据前面的真实输入进行下一个单词的输出进行训练，且第一个为<EOS>或者<SOS>，代表开始
target_data_input = [[2]+sentence[0:-1] for sentence in target_data]
print (target_data_input[0])
print (target_data[0])


# In[ ]:




