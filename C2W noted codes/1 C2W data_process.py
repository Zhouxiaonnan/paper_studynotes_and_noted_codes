#!/usr/bin/env python
# coding: utf-8

# # 数据处理模块

# ## 目录
# 
# * 数据集加载
# * 分句，分词以及划分数据集
# * 加载训练集
# * 构建word2id，char2id
# * 构建特征和标签
# * 生成torch数据导入类

# In[1]:


import  json
import nltk  # NLP tool kit


# In[2]:


# 数据集加载
datas = open("./data/wiki_00",encoding="utf-8").read().splitlines()
datas[0:5]


# In[3]:


json_data = json.loads(datas[0],strict=False)
json_data


# In[4]:


text_data = json_data["text"]
text_data


# In[5]:


text_data = text_data.replace("\n\n",". ")
text_data = text_data.replace("\n",". ")
text_data


# In[6]:


text_data = nltk.sent_tokenize(text_data)
text_data


# In[7]:


tokens = nltk.word_tokenize(text_data[-1])
tokens


# In[10]:


# 分句，分词以及划分数据集
num_words = 0
f_train = open("./data/train.txt","w",encoding="utf-8")
f_valid = open("./data/valid.txt","w",encoding="utf-8")
f_test = open("./data/test.txt","w",encoding="utf-8")
for i,data in enumerate(datas):
    if i%1000==0:
        print (i,len(datas))
    data = json.loads(data,strict=False)
    sentences = data["text"]
    sentences = sentences.replace("\n\n",". ")
    sentences = sentences.replace("\n",". ")
    sentences = nltk.sent_tokenize(sentences) # 分句
    for sentence in sentences:
        sentence = nltk.word_tokenize(sentence) # 分词
        if len(sentence)<10 or len(sentence)>100:
            continue
        num_words+=len(sentence)
        sentence = " ".join(sentence) +"\n"
        if num_words<=1000000:
            f_train.write(sentence)
        elif num_words<=1020000:
            f_valid.write(sentence)
        elif num_words<=1040000:
            f_test.write(sentence)
        else:
            break


# In[85]:


# 加载训练集
import numpy as np
datas = open("./data/train.txt",encoding="utf-8").read().strip("\n").splitlines()
datas = [s.split() for s in datas]
np.array(datas[0:5])


# In[86]:


# 构建word2id，char2id
from collections import Counter


# In[87]:


words = []
chars = []
for data in datas:
    for word in data:
        words.append(word.lower()) # 把所有词和所有字符保存下来，之后使用Counter进行统计
        chars.extend(word)
print (len(words))
print (len(chars))


# In[88]:


words = dict(Counter(words).most_common(5000-2)) # 仅保存前5000个左右个词，这里 -2 是因为还需要在词表前添加<pad>和<unk>
chars = dict(Counter(chars).most_common(512-3)) # 保存前500个左右的字符，这里 -3 是因为还需要在字符表前添加<pad>，<unk>，<start>
print (words)
print (chars)


# In[89]:


print (len(words))
print (len(chars))


# In[92]:


word2id = {"<pad>":0,"<unk>":1} # padding用数字为0，unknown单词为1
for word in words:
    word2id[word] = len(word2id)
word2id


# In[93]:


char2id = {"<pad>":0,"<unk>":1,"<start>":2}
for char in chars:
    char2id[char] = len(char2id)
char2id


# In[94]:


# 首先需要明白输入数据中训练特征和Labels是什么形状
# 输入特征是由词经过token之后的句子，并且要经过padding，其中句子中的每个词都被分成字符，并且也需要进行padding
# 因此是一个三维的输入，形状：sentence_num * max_sentence_length * max_character_length
# 模型通过前面的词预测后面的词，进而训练词向量，因此标签形状是：sentence_num * max_sentence_length

# 构建特征和标签
char_datas = [] # 输入特征characters_data
weights = [] # 对于词维度，因为我们不想训练padding0的部分，因此要生成一个weights，有单词为1，padding的部分为0
data = datas[0]
np.array(data)


# In[95]:


max_word_length = 16 # 这是指一个单词的最大长度
char_data = [[char2id["<start>"]]*max_word_length] # 第一个字符是start，标志句子的开始
char_data


# In[96]:


word = data[0]
word


# In[97]:


char_word = [] # 对于单词
for char in word:
    char_word.append(char2id.get(char,char2id["<unk>"])) # 如果能够找到char，则用char的id，如果找不到，则用<unk>的id
char_word # 这是一个单词的char编号


# In[98]:


char_word = char_word[0:max_word_length] + [char2id["<pad>"]]*(max_word_length-len(char_word)) # 对一个单词的char进行padding
char_word


# In[99]:


np.array(datas[0])


# In[100]:


data = datas[0]
for j,word in enumerate(data):
    char_word = []
    for char in word:
        char_word.append(char2id.get(char,char2id["<unk>"]))
    char_word = char_word[0:max_word_length] + [char2id["<pad>"]]*(max_word_length-len(char_word))
    data[j] = word2id.get(data[j].lower(),word2id["<unk>"])
    char_data.append(char_word)
print (data) # 这是将单词进行tokenize
print (char_data) # 这是将字符进行tokenize并且padding


# In[101]:


max_sentence_length = 100 # 这是一句话里最多有几个单词
weights.extend([1] * len(data)+[0]*(max_sentence_length-len(data))) # 得到weights
np.array(weights)


# In[102]:


max_sentence_length = 100
data = data[0:max_sentence_length]+[word2id["<pad>"]]*(max_sentence_length-len(data)) # 这是对句子进行Padding
np.array(data)


# In[103]:


# 因为句子进行了padding，因此padding0也需要在字符层面进行padding，也就整行都是0
char_data = char_data[0:max_sentence_length]+[[char2id["<pad>"]]*max_word_length]*(max_sentence_length-len(char_data)) 
np.array(char_data) # 最后一个句子的矩阵为 max_sentence_length（句子最大长度，即100个单词） * max_word_length（单词最大长度，即16个字符）


# In[104]:


# 对所有数据进行上述的整理
import numpy as np
datas = open("./data/train.txt",encoding="utf-8").read().strip("\n").splitlines()
datas = [s.split() for s in datas]
char_datas = []
weights = []
for i,data in enumerate(datas):
    if i%1000==0:
        print (i,len(datas))
    char_data = [[char2id["<start>"]]*max_word_length]
    for j,word in enumerate(data):
        char_word = []
        for char in word:
            char_word.append(char2id.get(char,char2id["<unk>"])) # 将字符转化成id
        char_word = char_word[0:max_word_length] +                     [char2id["<pad>"]]*(max_word_length-len(char_word))  # 将所有词pad成相同长度
        datas[i][j] = word2id.get(datas[i][j].lower(),word2id["<unk>"]) # 将词转化为id
        char_data.append(char_word)
    weights.extend([1] * len(datas[i])+[0]*(max_sentence_length-len(datas[i])))
    datas[i] = datas[i][0:max_sentence_length]+[word2id["<pad>"]]*(max_sentence_length-len(datas[i])) # 将所有句子pad成相同长度
    char_datas.append(char_data)
    char_datas[i] = char_datas[i][0:max_sentence_length]+                    [[char2id["<pad>"]]*max_word_length]*(max_sentence_length-len(char_datas[i])) # 将所有句子pad成相同长度

datas = np.array(datas)
char_datas = np.array(char_datas)
weights = np.array(weights)


# In[105]:


datas # 这是单词层面，形状为 sentence_num（句子个数） * max_sentence_length（最大句子长度）


# In[106]:


char_datas # 这是字符层面，形状为 sentence_num * max_sentence_length * max_word_length


# In[107]:


weights


# In[108]:


# reshape
datas = datas.reshape([-1])
char_datas = char_datas.reshape([-1,max_word_length])


# In[109]:


print (datas.shape)
print (char_datas.shape)
print (weights.shape)


# In[110]:


# 整理成python的class用于pytorch数据读入，见pycharm


# In[ ]:




