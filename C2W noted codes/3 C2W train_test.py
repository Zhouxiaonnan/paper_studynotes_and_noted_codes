#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
from model import C2W
from data import Char_LM_Dataset
from tqdm import tqdm
import config as argumentparser


# In[3]:


import config as argumentparser
config = argumentparser.ArgumentParser()


# In[5]:


if torch.cuda.is_available():
    torch.cuda.set_device(config.gpu)


# In[6]:


torch.cuda.is_available()


# In[7]:


# 导入训练集
training_set = Char_LM_Dataset(mode="train")
training_iter = torch.utils.data.DataLoader(dataset=training_set,
                                            batch_size=config.batch_size*config.max_sentence_length,
                                            shuffle=False,
                                            num_workers=2)


# In[8]:


# 导入验证集
valid_set = Char_LM_Dataset(mode="valid")
valid_iter = torch.utils.data.DataLoader(dataset=valid_set,
                                         batch_size=config.batch_size*config.max_sentence_length,
                                         shuffle=False,
                                         num_workers=0)


# In[9]:


# 导入测试集
test_set = Char_LM_Dataset(mode="test")
test_iter = torch.utils.data.DataLoader(dataset=test_set,
                                        batch_size=32*100,
                                        shuffle=False,
                                        num_workers=0)


# In[10]:


model = C2W(config)


# In[11]:


if config.cuda and torch.cuda.is_available():
    model.cuda()


# In[13]:


criterion = nn.CrossEntropyLoss(reduce=False)
optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
loss  = -1


# In[14]:


def get_test_result(data_iter,data_set):
    # 生成测试结果
    model.eval()
    all_ppl = 0
    for data, label,weights in data_iter:
        if config.cuda and torch.cuda.is_available():
            data = data.cuda()
            label = label.cuda()
            weights = weights.cuda()
        else:
            data = torch.autograd.Variable(data).long()
        label = torch.autograd.Variable(label).squeeze()
        out = model(data)
        loss_now = criterion(out, autograd.Variable(label.long()))
        ppl = (loss_now * weights.float()).view([-1, config.max_sentence_length]) # 困惑度
        ppl = torch.sum(ppl, dim=1) / torch.sum((weights.view([-1, config.max_sentence_length])) != 0, dim=1).float()
        ppl = torch.sum(torch.exp(ppl))
        all_ppl += ppl.data.item()
    return all_ppl*config.max_sentence_length/data_set.__len__()


# In[15]:


for epoch in range(config.epoch):
    model.train()
    process_bar = tqdm(training_iter)
    for data, label,weights in process_bar:
        if config.cuda and torch.cuda.is_available():
            data = data.cuda()
            label = label.cuda()
            weights = weights.cuda()
        else:
            data = torch.autograd.Variable(data).long()
        label = torch.autograd.Variable(label).squeeze()
        out = model(data)
        loss_now = criterion(out, autograd.Variable(label.long()))
        ppl = (loss_now*weights.float()).view([-1,config.max_sentence_length]) # pad不计算loss
        ppl = torch.sum(ppl,dim=1)/torch.sum((weights.view([-1,config.max_sentence_length]))!=0,dim=1).float()
        ppl = torch.mean(torch.exp(ppl)) # 计算困惑度
        loss_now = torch.sum(loss_now*weights.float())/torch.sum(weights!=0)
        if loss==-1:
            loss = loss_now.data.item()
        else:
            loss = 0.95 * loss + 0.05 * loss_now.data.item() # loss平滑输出
        process_bar.set_postfix(loss=loss,ppl=ppl.data.item())
        process_bar.update()
        optimizer.zero_grad()
        loss_now.backward()
        optimizer.step()
    print ("Valid ppl is:",get_test_result(valid_iter,valid_set))
    print ("Test ppl is:",get_test_result(test_iter,valid_set))


# In[ ]:




