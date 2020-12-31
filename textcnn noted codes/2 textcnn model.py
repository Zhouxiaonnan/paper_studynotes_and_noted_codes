#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn


# In[2]:


class BasicModule(nn.Module):
    def __init__(self):
        super(BasicModule, self).__init__()
        self.model_name = str(type(self))

    def load(self, path):
        self.load_state_dict(torch.load(path))

    def save(self, path):
        torch.save(self.state_dict(), path)

    def forward(self):
        pass


# In[3]:


import torch.nn.functional as F


# In[4]:


class TextCNN(BasicModule):

    def __init__(self, config):
        super(TextCNN, self).__init__()
        # 嵌入层
        # 使用预训练的词向量，freeze表示是否使用静态
        if config.embedding_pretrained is not None:
            self.embedding = nn.Embedding.from_pretrained(config.embedding_pretrained, freeze=False) 
            
        # 不使用预训练的词向量，在模型训练过程中优化词向量
        else:
            self.embedding = nn.Embedding(config.n_vocab, config.embed_size) # batchsize*l
            
        # 卷积层
        self.conv1d_1 = nn.Conv1d(config.embed_size, config.filter_num, config.filters[0])  # channel
        self.conv1d_2 = nn.Conv1d(config.embed_size, config.filter_num, config.filters[1])
        self.conv1d_3 = nn.Conv1d(config.embed_size, config.filter_num, config.filters[2])
        
        # 池化层
        self.Max_pool_1 = nn.MaxPool1d(config.sentence_max_size-3+1)
        self.Max_pool_2 = nn.MaxPool1d(config.sentence_max_size-4+1)
        self.Max_pool_3 = nn.MaxPool1d(config.sentence_max_size-5+1)
        
        # Dropout层
        self.dropout = nn.Dropout(config.dropout)
        
        #分类层
        self.fc = nn.Linear(config.filter_num*len(config.filters), config.label_num)
        
    def forward(self, x):
        x = x.long() 
        out = self.embedding(x) # bs * max_length * embedding_size
        out = out.transpose(1, 2).contiguous() # bs * embedding_size * length 需要将embedding的部分放在第一个维度
        x1 = F.relu(self.conv1d_1(out)) # convolution + relu
        x2 = F.relu(self.conv1d_2(out))
        x3 = F.relu(self.conv1d_3(out))
        x1 = self.Max_pool_1(x1).squeeze() # maxpooling  squeeze用于删除长度为1的维度
        x2 = self.Max_pool_2(x2).squeeze()
        x3 = self.Max_pool_3(x3).squeeze()
        print (x1.size(),x2.size(),x3.size())
        out = torch.cat([x1,x2,x3], 1) # concat
        out = self.dropout(out) # dropout
        out = self.fc(out) # 全连接层
        return out


# In[5]:


class config:
    def __init__(self):
        self.embedding_pretrained = None # 是否使用预训练的词向量
        self.n_vocab = 100 # 词表中单词的个数
        self.embed_size = 300 # 词向量的维度 
        self.cuda = False # 是否使用gpu
        self.filter_num = 100 # 每种尺寸卷积核的个数
        self.filters = [3,4,5] # 卷积核的尺寸
        self.label_num = 2 # 标签个数
        self.dropout = 0.5 # dropout的概率
        self.sentence_max_size = 50 #最大句子长度


# In[6]:


config = config()


# In[7]:


textcnn = TextCNN(config)


# In[8]:


from torchsummary import summary


# In[9]:


summary(textcnn, input_size=(50,))


# In[ ]:





# In[ ]:




