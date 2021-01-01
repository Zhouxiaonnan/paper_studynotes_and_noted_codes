#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
from model import CharTextCNN
from data import AG_Data
from tqdm import tqdm
import numpy as np


# In[ ]:


import config as argumentparser
config = argumentparser.ArgumentParser() # 读入参数设置
config.features = list(map(int,config.features.split(","))) # 将features用,分割，并且转成int
config.kernel_sizes = list(map(int,config.kernel_sizes.split(","))) # kernel_sizes,分割，并且转成int


# In[ ]:


if config.cuda and torch.cuda.is_available():  # 是否使用gpu
    torch.cuda.set_device(config.gpu)


# In[ ]:


torch.cuda.is_available() # 查看gpu是否可用


# In[ ]:


# 导入训练集
training_set = AG_Data(data_path="/AG/train.csv",l0=config.l0)
training_iter = torch.utils.data.DataLoader(dataset=training_set,
                                            batch_size=config.batch_size,
                                            shuffle=True,
                                            num_workers=0)


# In[ ]:


# 导入测试集
test_set = AG_Data(data_path="/AG/test.csv",l0=config.l0)

test_iter = torch.utils.data.DataLoader(dataset=test_set,
                                        batch_size=config.batch_size,
                                        shuffle=False,
                                        num_workers=0)


# In[ ]:


model = CharTextCNN(config) # 初始化模型


# In[ ]:


if config.cuda and torch.cuda.is_available(): # 如果使用gpu，将模型送进gpu
    model.cuda()


# In[ ]:


criterion = nn.CrossEntropyLoss() # 构建loss结构
optimizer = optim.Adam(model.parameters(), lr=config.learning_rate) #构建优化器
loss  = -1


# In[ ]:


def get_test_result(data_iter,data_set):
    # 生成测试结果
    model.eval()
    data_loss = 0
    true_sample_num = 0
    for data, label in data_iter:
        if config.cuda and torch.cuda.is_available():
            data = data.cuda()
            label = label.cuda()
        else:
            data = torch.autograd.Variable(data).float()
        out = model(data)
        true_sample_num += np.sum((torch.argmax(out, 1) == label).cpu().numpy()) # 得到一个batch的预测正确的样本个数
    acc = true_sample_num / data_set.__len__()
    return acc


# In[ ]:


for epoch in range(config.epoch):
    model.train()
    process_bar = tqdm(training_iter)
    for data, label in process_bar:
        if config.cuda and torch.cuda.is_available():
            data = data.cuda()  # 如果使用gpu，将数据送进gou
            label = label.cuda()
        else:
            data = torch.autograd.Variable(data).float()
        label = torch.autograd.Variable(label).squeeze()
        out = model(data)
        loss_now = criterion(out, autograd.Variable(label.long()))
        if loss == -1:
            loss = loss_now.data.item()
        else:
            loss = 0.95*loss+0.05*loss_now.data.item()  # 平滑操作
        process_bar.set_postfix(loss=loss_now.data.item()) # 输出loss，实时监测loss的大小
        process_bar.update()
        optimizer.zero_grad() # 梯度更新
        loss_now.backward()
        optimizer.step()
    


# In[ ]:


test_acc = get_test_result(test_iter, test_set)
print("The test acc is: %.5f" % test_acc)


# In[ ]:




