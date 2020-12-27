#!/usr/bin/env python
# coding: utf-8

# # Skip-Gram+层次Softmax数据处理模块

# In[1]:


import numpy as np
from collections import deque
from huffman_tree import HuffmanTree


# In[2]:


input_file = open("./data/text8.txt","r",encoding="utf-8")


# In[3]:


def _init_dict(min_count):
    word_count_sum = 0
    sentence_count = 0
    word2id_dict = dict()
    id2word_dict = dict()
    wordid_frequency_dict = dict()
    word_freq = dict()
    for line in input_file:
        line = line.strip().split()
        word_count_sum +=len(line)
        sentence_count +=1
        for i,word in enumerate(line):
            if i%1000000==0:
                print (i,len(line))
            if word_freq.get(word)==None:
                word_freq[word] = 1
            else:
                word_freq[word] += 1
    for i,word in enumerate(word_freq):
        if i % 100000 == 0:
            print(i, len(word_freq))
        if word_freq[word]<min_count:
            word_count_sum -= word_freq[word]
            continue
        word2id_dict[word] = len(word2id_dict)
        id2word_dict[len(id2word_dict)] = word
        wordid_frequency_dict[len(word2id_dict)-1] = word_freq[word]
    word_count =len(word2id_dict)
    return word2id_dict,id2word_dict,wordid_frequency_dict


# In[4]:


word2id_dict,id2word_dict,wordid_frequency_dict = _init_dict(20)


# In[7]:


len(word2id_dict)


# In[5]:


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


# In[6]:


wordId_list = get_wordId_list()


# In[8]:


wordId_list


# In[9]:


def get_batch_pairs(batch_size,window_size,index,word_pairs_queue):
    while len(word_pairs_queue) < batch_size:
        for _ in range(1000):
            if index == len(wordId_list):
                index = 0
            for i in range(max(index - window_size, 0), min(index + window_size + 1,len(wordId_list))):
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


# In[10]:


index = 0
word_pairs_queue = deque()
result_pairs = get_batch_pairs(32,3,index,word_pairs_queue)


# In[11]:


result_pairs


# In[12]:


huffman_tree = HuffmanTree(wordid_frequency_dict)  # 霍夫曼树

# huffman_pos_path：从根节点到叶节点要向左走的节点id
# huffman_neg_path：从根节点到叶节点要向右走的节点id
huffman_pos_path, huffman_neg_path = huffman_tree.get_all_pos_and_neg_path() 

# pos_pairs：样本语料中存在的真实单词对
def get_pairs(pos_pairs):
    neg_word_pair = []
    pos_word_pair = []
    for pair in pos_pairs: # 每一个单词对
        
        # pos_word_pair：从根节点走向单词对中的第二个元素（周围词）需要向左走的内部节点和根节点的id
        # neg_word_pair：从根节点走向单词对中的第二个元素（周围词）需要向右走的内部节点和根节点的id
        pos_word_pair += zip([pair[0]] * len(huffman_pos_path[pair[1]]), huffman_pos_path[pair[1]]) 
        neg_word_pair += zip([pair[0]] * len(huffman_neg_path[pair[1]]), huffman_neg_path[pair[1]])
    return pos_word_pair, neg_word_pair


# 在pos_word_pair中，如果有(3, 100)，说明对于wordid为3的中间词，要走到某个周围词需要在id为100的内部节点上往左走
# 在neg_word_pair中，如果有(4, 200)，说明对于wordid为4的中间词，要走到某个周围词需要在id为200的内部节点上往右走


# In[14]:


pos_word_pair, neg_word_pair = get_pairs(result_pairs)


# In[15]:


pos_word_pair


# In[16]:


neg_word_pair


# In[ ]:




