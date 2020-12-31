#!/usr/bin/env python
# coding: utf-8

# In[1]:


from pytorchtools import EarlyStopping
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
from model import TextCNN
from data import MR_Dataset
import numpy as np


# In[2]:


import config as argumentparser
config = argumentparser.ArgumentParser()
config.filters = list(map(int,config.filters.split(",")))


# In[3]:


torch.manual_seed(config.seed)


# In[4]:


if torch.cuda.is_available():
    torch.cuda.set_device(config.gpu)


# In[5]:


torch.cuda.is_available()


# In[6]:


i=0


# In[7]:


early_stopping = EarlyStopping(patience=10, verbose=True,cv_index=i) # early stopping设置


# In[8]:


training_set = MR_Dataset(state="train",k=i) # 每次一个batch
config.n_vocab = training_set.n_vocab
training_iter = torch.utils.data.DataLoader(dataset=training_set,
                                            batch_size=config.batch_size,
                                            shuffle=True,
                                            num_workers=2)


# In[9]:


if config.use_pretrained_embed: # 是否使用预训练的词向量
    config.embedding_pretrained = torch.from_numpy(training_set.weight).float()
else:
    pass


# In[10]:


valid_set = MR_Dataset(state="valid", k=i) # 验证集
valid_iter = torch.utils.data.DataLoader(dataset=valid_set,
                                         batch_size=config.batch_size,
                                         shuffle=False,
                                         num_workers=2)


# In[11]:


test_set = MR_Dataset(state="test", k=i) # 测试集
test_iter = torch.utils.data.DataLoader(dataset=test_set,
                                        batch_size=config.batch_size,
                                        shuffle=False,
                                        num_workers=2)


# In[12]:


model = TextCNN(config)


# In[13]:


if config.cuda and torch.cuda.is_available():
    model.cuda()
    config.embedding_pretrained.cuda()


# In[14]:


criterion = nn.CrossEntropyLoss() # loss
optimizer = optim.Adam(model.parameters(), lr=config.learning_rate) # optimizer
count = 0
loss_sum = 0


# In[15]:


def get_test_result(data_iter,data_set):
    model.eval() # 使用全部的神经元
    data_loss = 0
    true_sample_num = 0
    for data, label in data_iter:
        if config.cuda and torch.cuda.is_available():
            data = data.cuda()
            label = label.cuda()
        else:
            data = torch.autograd.Variable(data).long()
        out = model(data)
        loss = criterion(out, autograd.Variable(label.long()))
        data_loss += loss.data.item()
        true_sample_num += np.sum((torch.argmax(out, 1) == label).cpu().numpy()) #(0,0.5)
    acc = true_sample_num / data_set.__len__()
    return data_loss,acc


# In[16]:


for epoch in range(config.epoch):
    # 训练开始
    model.train() # dropout层被激活
    for data, label in training_iter:
        if config.cuda and torch.cuda.is_available():
            data = data.cuda()
            label = label.cuda()
        else:
            data = torch.autograd.Variable(data).long()
        label = torch.autograd.Variable(label).squeeze()
        out = model(data)
        # l2_alpha*w^2
        l2_loss = config.l2*torch.sum(torch.pow(list(model.parameters())[1],2)) # L2正则
        loss = criterion(out, autograd.Variable(label.long()))+l2_loss # 原loss加上L2正则项
        loss_sum += loss.data.item()
        count += 1
        if count % 100 == 0:
            print("epoch", epoch, end='  ')
            print("The loss is: %.5f" % (loss_sum / 100))
            loss_sum = 0
            count = 0
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # save the model in every epoch
    # 一轮训练结束
    # 验证集上测试
    valid_loss,valid_acc = get_test_result(valid_iter,valid_set)
    early_stopping(valid_loss, model)
    print ("The valid acc is: %.5f" % valid_acc)
    if early_stopping.early_stop:
        print("Early stopping")
        break
# 训练结束，开始测试
model.load_state_dict(torch.load('./checkpoints/checkpoint%d.pt'%i))
test_loss, test_acc = get_test_result(test_iter, test_set)
print("The test acc is: %.5f" % test_acc)


# In[33]:


x = "it's so bad"
x = x.split()


# In[34]:


x = [training_set.word2id[word] for word in x]
x = np.array(x+[0]*(59-len(x))).reshape([1,-1])
x


# In[38]:


x = torch.autograd.Variable(torch.Tensor(x)).long()
out = model(x)


# In[39]:


out


# In[ ]:




