#!/usr/bin/env python
# coding: utf-8

# In[2]:


import torch
import torch.nn as nn
import numpy as np


# In[ ]:


# batch normalization1
# weights, 
# 训练 测试
# 0.95*prev_mean+0.05*now_mean
# 0.95*prev_std+0.05*now_std


# In[3]:


class CharTextCNN(nn.Module):
    def __init__(self,config):
        super(CharTextCNN,self).__init__()
        in_features = [config.char_num] + config.features[0:-1] # features不算最后一个
        out_features = config.features # 
        kernel_sizes = config.kernel_sizes # 卷积核尺寸
        self.convs = []
        self.conv1 = nn.Sequential(
                    nn.Conv1d(in_features[0], # 输入的features个数
                              out_features[0], # 输出的features个数
                              kernel_size=kernel_sizes[0], # 卷积核尺寸
                              stride=1 # 步长
                             ), # 一维卷积
                    nn.BatchNorm1d(out_features[0]), # bn层，输入的features个数是上一层conv1d层的输出
                    nn.ReLU(), # relu激活函数层
                    nn.MaxPool1d(kernel_size=3, # maxpool的尺寸
                                 stride=3 # 步长（即没有overlap）
                                ) #一维池化层
                ) # 卷积 + bn + relu + pooling 模块
        self.conv2  = nn.Sequential(
            nn.Conv1d(in_features[1], out_features[1], kernel_size=kernel_sizes[1], stride=1),
            nn.BatchNorm1d(out_features[1]),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=3)
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(in_features[2], out_features[2], kernel_size=kernel_sizes[2], stride=1),
            nn.BatchNorm1d(out_features[2]),
            nn.ReLU()
        )
        self.conv4 = nn.Sequential(
            nn.Conv1d(in_features[3], out_features[3], kernel_size=kernel_sizes[3], stride=1),
            nn.BatchNorm1d(out_features[3]),
            nn.ReLU()
        )
        self.conv5 = nn.Sequential(
            nn.Conv1d(in_features[4], out_features[4], kernel_size=kernel_sizes[4], stride=1),
            nn.BatchNorm1d(out_features[4]),
            nn.ReLU()
        )
        self.conv6 = nn.Sequential(
            nn.Conv1d(in_features[5], out_features[5], kernel_size=kernel_sizes[5], stride=1),
            nn.BatchNorm1d(out_features[5]),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=3)
        )
        self.fc1 = nn.Sequential(
            nn.Linear(8704, 1024), # 全连接层 # (1014(L0) - 96) / 27 * 256
            nn.ReLU(),
            nn.Dropout(p=config.dropout) # dropout层
        ) # 全连接+relu+dropout模块

        self.fc2 = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(p=config.dropout)
        )

        self.fc3 = nn.Linear(1024, config.num_classes)
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)

        x = x.view(x.size(0), -1) # 变成二维送进全连接层
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x


# In[3]:





# In[9]:


class config:
    def __init__(self):
        self.char_num = 70  # 字符表中字符的个数
        self.features = [256,256,256,256,256,256] # 每一层特征个数（feature_maps的个数）
        self.kernel_sizes = [7,7,3,3,3,3] # 每一层的卷积核尺寸
        self.dropout = 0.5 # dropout概率
        self.num_classes = 4 # 数据的类别个数


# In[10]:


config = config()
chartextcnn = CharTextCNN(config)
test = torch.zeros([64,70,1014])
out = chartextcnn(test)


# In[11]:


out


# In[12]:


out.shape


# In[13]:


from torchsummary import summary


# In[15]:


summary(chartextcnn, input_size=(70,1014))
# tensorflow: bn: 256*4


# In[ ]:




