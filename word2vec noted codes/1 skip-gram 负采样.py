#!/usr/bin/env python
# coding: utf-8

# # Skip-Gram+NGE数据处理模块

# 
# **目录：**
# 1. 词表构建
# 
# 2. 正样本生成
# 
# 3. 负样本采样
# 
# 4. 负样本生成
# 
# ---

# In[20]:


import numpy as np
from collections import deque


# In[40]:


input_file = open("./data/text8.txt","r",encoding="utf-8")


# In[41]:


def _init_dict(min_count):
    word_count_sum = 0 # 总字数
    sentence_count = 0 # 句子数
    word2id_dict = dict()
    id2word_dict = dict()
    wordid_frequency_dict = dict() # wordid对应该word出现频率的字典
    word_freq = dict() # 每个单词出现的频率
    for line in input_file:
        line = line.strip().split()
        word_count_sum += len(line) # 加单个句子长度（计算总字数）
        sentence_count += 1 # 计算句子数
        for i,word in enumerate(line): # 每个单词
            if i%1000000==0:
                print (i,len(line))
            if word_freq.get(word) == None: # 如果单词频率字典中还没有这个单词，则为1
                word_freq[word] = 1
            else: # 否则 + 1
                word_freq[word] += 1
    for i,word in enumerate(word_freq): # 遍历每个单词及其频率
        if i % 100000 == 0:
            print(i, len(word_freq))
        if word_freq[word] < min_count: # 如果频率小于一个阈值则删掉
            word_count_sum -= word_freq[word] # 总字数 - 该词出现频率
            continue
        word2id_dict[word] = len(word2id_dict) 
        id2word_dict[len(id2word_dict)] = word
        wordid_frequency_dict[len(word2id_dict) - 1] = word_freq[word] 
    word_count =len(word2id_dict)
    return word2id_dict,id2word_dict,wordid_frequency_dict


# In[42]:


word2id_dict,id2word_dict,wordid_frequency_dict = _init_dict(3)


# In[43]:


word2id_dict


# In[44]:


# 将语料转换成id的形式
def get_wordId_list():
    input_file = open("./data/text8.txt", encoding="utf-8")
    sentence = input_file.readline()
    wordId_list = []  # 一句中的所有word 对应的 id
    sentence = sentence.strip().split(' ')
    for i,word in enumerate(sentence):
        if i%1000000==0:
            print (i,len(sentence))
        try:
            word_id = word2id_dict[word]
            wordId_list.append(word_id)
        except:
            continue
    return wordId_list


# In[45]:


wordId_list = get_wordId_list()


# In[46]:


wordId_list


# In[47]:


# 得到正样本的pairs
def get_batch_pairs(batch_size,window_size,index,word_pairs_queue):
    while len(word_pairs_queue) < batch_size:
        if index==len(wordId_list):
            index = 0
        for _ in range(1000):
            for i in range(max(index - window_size, 0),min(index + window_size + 1,len(wordId_list))):
                wordId_w = wordId_list[index]
                wordId_v = wordId_list[i]
                if index == i:  # 上下文=中心词 跳过
                    continue
                word_pairs_queue.append((wordId_w, wordId_v))
            index+=1
    result_pairs = []  # 返回mini-batch大小的正采样对
    for _ in range(batch_size):
        result_pairs.append(word_pairs_queue.popleft())
    return result_pairs


# In[48]:


index = 0
word_pairs_queue = deque()
result_pairs = get_batch_pairs(32,3,index,word_pairs_queue)


# In[49]:


result_pairs


# In[52]:


# 生成采样表
def _init_sample_table():
    sample_table = []
    sample_table_size = 1e8 # 采样表的长度，即1e8个单词
    pow_frequency = np.array(list(wordid_frequency_dict.values())) ** 0.75 # 计算每个单词的采样频率
    word_pow_sum = sum(pow_frequency) # 归一化
    ratio_array = pow_frequency / word_pow_sum
    word_count_list = np.round(ratio_array * sample_table_size)
    for word_index, word_freq in enumerate(word_count_list): # 得到采样表，之后从采样表中进行随机选择，即得到采样
        sample_table += [word_index] * int(word_freq)
    sample_table = np.array(sample_table)
    return sample_table


# In[53]:


sample_table = _init_sample_table()


# In[54]:


sample_table


# In[55]:


# 从采样表中随机选择，得到采样
def get_negative_sampling(positive_pairs, neg_count):
    neg_v = np.random.choice(sample_table, size=(len(positive_pairs), neg_count)).tolist()
    return neg_v


# In[8]:


# import numpy as np
# np.random.choice([1,1,1,1,2,2,2,2,3,3,3,4,4,5,6,7,8,8,8,8,9,9,9], size = (4,3))


# In[9]:


neg_v = get_negative_sampling(result_pairs,3)


# In[57]:


neg_v

