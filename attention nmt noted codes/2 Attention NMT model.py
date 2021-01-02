#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import numpy as np
from torch.nn import functional as F
from torch.autograd import Variable


# In[6]:


class Attention_NMT(nn.Module):
    def __init__(self,source_vocab_size,target_vocab_size,embedding_size,
                 source_length,target_length,lstm_size,batch_size = 32):
        super(Attention_NMT,self).__init__()
        self.source_embedding =nn.Embedding(source_vocab_size,embedding_size) # source_vocab_size * embedding_size
        self.target_embedding = nn.Embedding(target_vocab_size,embedding_size) # target_vocab_size * embedding_size
        
        # if batch_first==False: input_shape=[length,batch_size,embedding_size]
        # bidirectional = True 双向LSTM，decoder不需要双向
        self.encoder = nn.LSTM(input_size=embedding_size,hidden_size=lstm_size,num_layers=1,
                               bidirectional=True,batch_first=True) 
        self.decoder = nn.LSTM(input_size = embedding_size + 2 * lstm_size, hidden_size=lstm_size,num_layers=1,
                               batch_first=True)
        self.attention_fc_1 = nn.Linear(3 * lstm_size, 3 * lstm_size) # 注意力机制全连接层1
        self.attention_fc_2 = nn.Linear(3 * lstm_size, 1) # 注意力机制全连接层2
        self.class_fc_1 = nn.Linear(embedding_size + 2 * lstm_size + lstm_size, 2 * lstm_size) # 分类全连接层1
        self.class_fc_2 = nn.Linear(2 * lstm_size, target_vocab_size) # 分类全连接层2
        
    def attention_forward(self,input_embedding,dec_prev_hidden,enc_output):
        
        # query
        # batch_size * 100 * lstm_size
        # 之所以要复制100份，是因为 query 需要与每个 key 进行 concat，key 的数量就是 max_sentence_length = 100
        prev_dec_h = dec_prev_hidden[0].squeeze().unsqueeze(1).repeat(1, 100, 1) 
        
        # query 与 key 进行 concat
        # qeury 的最后一个维度长度为 lstm_size，因为 encoder 是双向 LSTM，所以 key 的最后一个维度长度为 2 * lstm_size
        # batch_size * max_sentence_length * (3 * lstm_size)
        atten_input = torch.cat([enc_output, prev_dec_h], dim=-1)
        
        # 全连接层，得到eij
        # batch_size * max_sentence_length * 1
        attention_weights = self.attention_fc_2(F.relu(self.attention_fc_1(atten_input))) 
        
        # softmax 层，得到 alphaij
        # batch_size * max_sentence_length * 1
        # 第三个维度代表每个输入单词对应的 hidden state 的权重，即当前 decoder 时间步需要从这些 hidden state 中抽取多少信息
        attention_weights = F.softmax(attention_weights, dim=1)
        
        # 加权平均，在单词的维度上进行了相加
        # batch_size * 1 * (2 * lstm_size)
        atten_output = torch.sum(attention_weights * enc_output, dim=1).unsqueeze(1) 
        
        # input_embedding 是这一时间步的单词输入（即前一个时间步的预测单词的embedding）
        # batch_size * 1 * (embedding_size + 2 * lstm_size)
        dec_lstm_input = torch.cat([input_embedding, atten_output], dim=2) 
        
        # decoder 一个时间步的输入是：
        # dec_lstm_input 经过 attention 层计算得到的向量（ci）
        # decoder 前一步的 hidden state (s(i-1))
        dec_output, dec_hidden = self.decoder(dec_lstm_input, dec_prev_hidden)
        
        # dec_output: batch_size * 1 * lstm_size，预测单词的词向量
        # dec_hidden: [batch_size * 1 * lstm_size, batch_size * 1 * lstm_size]，当前时间步的 hidden state
        return atten_output, dec_output, dec_hidden
    
    def forward(self, source_data,target_data, mode = "train",is_gpu=True):
        source_data_embedding = self.source_embedding(source_data) # batch_size * max_sentence_length * embedding_size
        
        # enc_output.shape: batch_size * max_sentence_length * (2 * lstm_size) 
        # 为什么最后一个维度是 2 * lstm_size 呢，因为采用的是双向LSTM，最后是将 hidden state 进行了 concat
        # enc_hidden：[[h1,h2],[c1,c2]] 返回每个方向最后一个时间步的h和c
        enc_output, enc_hidden = self.encoder(source_data_embedding)
        
        # 用于存储 attention outputs 和 decoder outpouts
        self.atten_outputs = Variable(torch.zeros(target_data.shape[0],
                                                  target_data.shape[1],
                                                  enc_output.shape[2])) # batch_size * max_sentence_length * (2 * lstm_size)
        self.dec_outputs = Variable(torch.zeros(target_data.shape[0],
                                                target_data.shape[1],
                                                enc_hidden[0].shape[2])) # batch_size * max_sentence_length * lstm_size
        if is_gpu:
            self.atten_outputs = self.atten_outputs.cuda()
            self.dec_outputs = self.dec_outputs.cuda()
        # enc_output: bs*length*(2*lstm_size)
        if mode=="train": 
            target_data_embedding = self.target_embedding(target_data) # batch_size * max_sentence_length * embedding_size
            
            # [h1, c1]
            # dec_prev_hidden[0]: 1 * bs * lstm_size, dec_prev_hidden[1]: 1 * bs * lstm_size
            # unsqueeze(0) 指在第一个地方加上一个维度
            dec_prev_hidden = [enc_hidden[0][0].unsqueeze(0),enc_hidden[1][0].unsqueeze(0)] 
            
            for i in range(100): # 每一个时间步
                
                # bs * 1 * embedding_size 每个单词对应的 embedding
                # 这里的 input_embedding 代表前一个时间步的单词预测的词向量，第一个输入是 <EOS>
                input_embedding = target_data_embedding[:,i,:].unsqueeze(1)
                
                # 输入attention层
                atten_output, dec_output, dec_hidden = self.attention_forward(input_embedding,
                                                                              dec_prev_hidden,
                                                                              enc_output)
                
                # 保存 attention 层加权平均后的输出
                self.atten_outputs[:,i] = atten_output.squeeze()
                
                # 保存 decoder 每一个时间步的输出
                self.dec_outputs[:,i] = dec_output.squeeze()
                
                # 将当前时间步的 hidden state 作为下一个时间步的输入
                dec_prev_hidden = dec_hidden
            
            # concat
            # 使用当前单词的词向量，注意力机制的输出，以及 decoder 的 hidden state
            # bs * max_sentence_length * (embedding_size + 2 * lstm_size + lstm_size)
            class_input = torch.cat([target_data_embedding,self.atten_outputs,self.dec_outputs],dim=2) 
            
            # 全连接层
            outs = self.class_fc_2(F.relu(self.class_fc_1(class_input)))
            
        else: # 如果是预测
            input_embedding = self.target_embedding(target_data) # embedding
            dec_prev_hidden = [enc_hidden[0][0].unsqueeze(0),enc_hidden[1][0].unsqueeze(0)] # 前一个时间步的 hidden_state
            outs = []
            for i in range(100):
                
                # attention
                atten_output, dec_output, dec_hidden = self.attention_forward(input_embedding,
                                                                              dec_prev_hidden,
                                                                              enc_output)

                # concat
                class_input = torch.cat([input_embedding,atten_output,dec_output],dim=2)
                
                # 全连接层
                pred = self.class_fc_2(F.relu(self.class_fc_1(class_input)))
                
                # 预测的单词
                pred = torch.argmax(pred,dim=-1)
                outs.append(pred.squeeze().cpu().numpy()) # 记录下预测的单词
                
                # 往前走一步
                dec_prev_hidden = dec_hidden
                
                # 使用预测的单词进行 embedding，作为下一个时间步的输入单词
                input_embedding = self.target_embedding(pred)
                
        return outs


# In[7]:


deep_nmt = Attention_NMT(source_vocab_size=30000,target_vocab_size=30000,embedding_size=256,
                 source_length=100,target_length=100,lstm_size=256, batch_size=32)
source_data = torch.Tensor(np.zeros([64,100])).long()
target_data = torch.Tensor(np.zeros([64,100])).long()
preds = deep_nmt(source_data,target_data,is_gpu=False)
print (preds.shape)
target_data = torch.Tensor(np.zeros([64, 1])).long()
preds = deep_nmt(source_data, target_data,mode="test",is_gpu=False)
print(np.array(preds).shape)


# In[ ]:




