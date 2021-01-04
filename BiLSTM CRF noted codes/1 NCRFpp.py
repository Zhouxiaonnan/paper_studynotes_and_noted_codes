#!/usr/bin/env python
# coding: utf-8

# ### 1. NCRFpp 数据格式 
# #### github： https://github.com/jiesutd/NCRFpp  
# ###  1.1 NCRF++ supports both BIO and BIOES(BMES) tag scheme.   
# ###  1.2 You can refer the data format in sample_data.

# In[1]:


get_ipython().system('ls ./NCRFpp/sample_data/')


# In[2]:


get_ipython().system('cat -n ./NCRFpp/sample_data/dev.bmes | head -n 30 ')


# In[3]:


get_ipython().system('cat -n ./NCRFpp/sample_data/dev.cappos.bmes | head -n 20 ')


# #### ==========================================================================================
# #### 对比dev.bmes与dev.cappos.bmes的不同，可看出相关特征如何加入数据中
# #### 参考sample_data里的格式将CoNLL-2003转化为NCRFpp支持的格式。   
# #### 这里我们不处理特征与POS等，只用字符本身
# #### 下面下载CoNLL-2003数据，进行转化

# In[4]:


import os
RAW_DATA_PATH = './NCRFpp/data/CoNLL-2003'
SIMPLE_DATA_PATH = './NCRFpp/data/conll_2003_simple/'


# In[5]:


get_ipython().system('cat -n ./NCRFpp/data/CoNLL-2003/dev.txt | head -n 20 ')


# In[6]:


for file_type in ['train', 'dev', 'test']:
    raw_file_path = os.path.join(RAW_DATA_PATH, file_type + '.txt')
    out_file_path = os.path.join(SIMPLE_DATA_PATH, file_type + '.bmes')
    content = ''
    with open(raw_file_path, 'r') as f:
        lines = []
        for line in f.readlines():
            line = line.strip()
            if len(line) == 0:
                lines.append(line)
            else:
                temp = line.split(' ')
                lines.append(' '.join([temp[0], temp[-1]]))
        content = '\n'.join(lines)
    with open(out_file_path, 'w') as f:
        f.write(content)   
    print(raw_file_path, out_file_path)


# In[7]:


get_ipython().system('cat -n ./NCRFpp/data/conll_2003_simple/dev.bmes | head -n 20 ')


# ### 2. Pretrained Word Embedding
# #### 2.1 查看NCRFpp支持的Word Embedding格式
# #### 2.2 download page: https://nlp.stanford.edu/projects/glove/

# In[8]:


get_ipython().system('cat -n NCRFpp/sample_data/sample.word.emb | head -n 3')


# In[10]:


# 下载glove.6B.zip
get_ipython().system('wget http://downloads.cs.stanford.edu/nlp/data/glove.6B.zip')


# In[9]:


# 解压文件，将glove.6B.100d.txt移动到/data下
get_ipython().system('cat -n ./NCRFpp/data/glove.6B.100d.txt | head -n 3')
# NCRFpp/data/glove.6B.100d.txt可作为预训练词向量使用

