#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import numpy as np


# In[10]:


class C2W(nn.Module):
    def __init__(self, config):
        super(C2W, self).__init__()
        self.char_hidden_size = config.char_hidden_size
        self.word_embed_size = config.word_embed_size
        self.lm_hidden_size = config.lm_hidden_size
        self.character_embedding = nn.Embedding(config.n_chars,config.char_embed_size) # 字符嵌入层
        self.sentence_length = config.max_sentence_length
        self.char_lstm = nn.LSTM(input_size=config.char_embed_size,hidden_size=config.char_hidden_size, # 双向的LSTM
                            bidirectional=True,batch_first=True)  # 字符lstm
        self.lm_lstm = nn.LSTM(input_size=self.word_embed_size,hidden_size=config.lm_hidden_size,batch_first=True) # 语言模型lstm
        self.fc_1 = nn.Linear(2*config.char_hidden_size,config.word_embed_size) # 线性组合生成词表示
        self.fc_2 =nn.Linear(config.lm_hidden_size,config.vocab_size) # 生成类别用于预测

    def forward(self, x):
        x = torch.Tensor(x).long()
        
        # 因为输入数据进行过reshape，所以第一个维度的长度为batch_size * max_sentence_length，代表每个单词
        # 第二个维度代表单词的字符，这里embedding就是对每个单词的字符进行embedding
        # [(bs * max_sentence_length) * max_word_length] -> 
        # [(bs * max_sentence_length) * max_word_length * embedding_size]
        input = self.character_embedding(x)
        
        # [(bs * max_sentence_length) * max_word_length * embedding_size] -> 
        # [(bs * max_sentence_length) * max_word_length * (hidden_size * 2)]
        char_lstm_result = self.char_lstm(input)
        
        # 正向的最后一个（即句子的最后一个单词）与反向的第一个（即句子的最后一个单词）进行concat
        word_input = torch.cat([char_lstm_result[0][:,-1,0:self.char_hidden_size],
                                char_lstm_result[0][:,0,self.char_hidden_size:]],dim=1)
        word_input = self.fc_1(word_input) # concat后输入全连接层，得到[(bs * max_sentence_length) * word_embedding]（得到词向量）
        word_input = word_input.view([-1,self.sentence_length,self.word_embed_size]) # 转换成三维的[bs * msl * word_embedding]
        lm_lstm_result = self.lm_lstm(word_input)[0].contiguous() # 输入语言模型，得到[bs * msl * hidden_size]
        lm_lstm_result = lm_lstm_result.view([-1,self.lm_hidden_size]) # 转换成[(bs * msl) * hidden_size]
        out = self.fc_2(lm_lstm_result) # 分类[(bs * msl) * vocabulary_size]
        return out


# In[11]:


class config:
    def __init__(self):
        self.n_chars = 64  # 字符的个数
        self.char_embed_size = 50 # 字符嵌入大小
        self.max_sentence_length = 8 # 最大句子长度
        self.char_hidden_size = 50 # 字符lstm的隐藏层神经元个数
        self.lm_hidden_size = 150 # 语言模型的隐藏神经元个数
        self.word_embed_size = 50 # 生成的词表示大小
        config.vocab_size = 1000 # 词表大小


# In[12]:


config = config()
c2w = C2W(config)
test = np.zeros([64,16])
out = c2w(test)


# In[16]:


out


# In[17]:


out.shape


# In[18]:


print (c2w)


# In[ ]:




