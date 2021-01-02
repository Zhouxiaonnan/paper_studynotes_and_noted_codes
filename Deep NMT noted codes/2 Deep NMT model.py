#!/usr/bin/env python
# coding: utf-8

# In[2]:


import torch
import torch.nn as nn
import numpy as np


# In[3]:


class Deep_NMT(nn.Module):
    def __init__(self,source_vocab_size,target_vocab_size,embedding_size,
                 source_length,target_length,lstm_size):
        super(Deep_NMT,self).__init__()
        
        # source_vocab_size 原语言词表大小
        # embedding_size 词向量长度
        self.source_embedding =nn.Embedding(source_vocab_size,embedding_size)  # batch_size * source_vocab_size * embedding_size
        self.target_embedding = nn.Embedding(target_vocab_size,embedding_size) # batch_size * target_vocab_size * embedding_size
        self.encoder = nn.LSTM(input_size=embedding_size,hidden_size=lstm_size,num_layers=4, # LSTM的深度为4层
                               batch_first=True) # if batch_first==False: input_shape=[length,batch_size,embedding_size]
        self.decoder = nn.LSTM(input_size=embedding_size,hidden_size=lstm_size,num_layers=4,
                               batch_first=True)
        self.fc = nn.Linear(lstm_size, target_vocab_size) # 全连接层
    def forward(self, source_data,target_data, mode = "train"):
        source_data_embedding = self.source_embedding(source_data) # batch_size * max_sentence_length * embedding_size
        
        # enc_output.shape: batch_size * max_sentence_length * lstm_size 只返回最高层的所有hidden
        # enc_hidden：[[h1,h2,h3,h4],[c1,c2,c3,c4]] 返回每层最后一个时间步的h和c
        enc_output, enc_hidden = self.encoder(source_data_embedding)
        
        # 如果是'train'，那么在decoder中每一步需要输入真实的序列，以计算Loss训练模型
        if mode=="train": 
            target_data_embedding = self.target_embedding(target_data) # batch_size * max_sentence_length * embedding_size
            
            # dec_output.shape: batch_size * max_sentence_length * lstm_size 只返回最高层的所有hidden
            # dec_hidden：[[h1,h2,h3,h4],[c1,c2,c3,c4]] 返回每层最后一个时间步的h和c
            # 输入使用了encoder的enc_hidden输出
            dec_output, dec_hidden = self.decoder(target_data_embedding,enc_hidden)

            outs = self.fc(dec_output) # batch_size * max_sentence_length * target_vocab_size
        
        # 如果是'test'，在decoder中，使用上一步的预测作为下一步的输入
        else:
            target_data_embedding = self.target_embedding(target_data) # batch_size * 1 * embedding_size 第二个维度长度为1是因为一次输入一个样本
            dec_prev_hidden = enc_hidden # [[h1,h2,h3,h4],[c1,c2,c3,c4]] decoder的输入为enc_hidden
            outs = []
            for i in range(100): # 将输出序列的长度限制为100
                
                # dec_output.shape: batch_size * 1 * lstm_size 只返回最高层的所有hidden
                # dec_hidden：[[h1,h2,h3,h4],[c1,c2,c3,c4]] 返回每层最后一个时间步的h和c                
                dec_output, dec_hidden = self.decoder(target_data_embedding, dec_prev_hidden)
                pred = self.fc(dec_output) # batch_size * 1 * target_vocab_size 将最高层的所有hidden输入全连接层
                pred = torch.argmax(pred,dim=-1) # batch_size * 1 最大的对应单词为该cell的输出单词
                outs.append(pred.squeeze().cpu().numpy()) # 加入输出单词
                dec_prev_hidden = dec_hidden # [[h1,h2,h3,h4],[c1,c2,c3,c4]] 将之前dec_hidden输入到下一个cell中
                target_data_embedding = self.target_embedding(pred) # batch_size * 1 * embedding_size 将之前的预测输入到下一个cell中
                
        return outs


# In[4]:


deep_nmt = Deep_NMT(source_vocab_size=30000,target_vocab_size=30000,embedding_size=256,
                 source_length=100,target_length=100,lstm_size=256)
source_data = torch.Tensor(np.zeros([64,100])).long()
target_data = torch.Tensor(np.zeros([64,100])).long()
preds = deep_nmt(source_data,target_data)
print (preds.shape)
target_data = torch.Tensor(np.zeros([64, 1])).long()
preds = deep_nmt(source_data, target_data,mode="test")
print(np.array(preds).shape)


# In[ ]:




