#!/usr/bin/env python
# coding: utf-8

# # Huffman树的构建

# In[28]:


class HuffmanNode:
    def __init__(self,word_id,frequency):
        self.word_id = word_id  # wordid
        self.frequency = frequency # 单词出现频率
        self.left_child = None # 左
        self.right_child = None # 右
        self.father = None # 父结点
        self.Huffman_code = [] # 往左为1，往右为0，表示从根节点到叶节点需要经过往左往右的选择的序列
        self.path = [] # 表示从根节点到叶节点需要经过的内部节点的编号的序列


# In[29]:


wordid_frequency_dict = {0: 4, 1: 6, 2: 3, 3: 2, 4: 2} # wordid对应的单词的出现频率，为每个叶节点的权重
wordid_code = dict()
wordid_path = dict()
unmerge_node_list = [HuffmanNode(wordid,frequency) for wordid,frequency in wordid_frequency_dict.items()] # 生成节点
huffman = [HuffmanNode(wordid,frequency) for wordid,frequency in wordid_frequency_dict.items()]


# In[30]:


# Huffman树的节点merge函数
def merge_node(node1,node2):
    sum_frequency = node1.frequency + node2.frequency # 两个节点的频率相加，为他们的父节点的权重
    mid_node_id = len(huffman) # 两个节点的父结点的index，这个index是从词表最后一个id开始
    father_node = HuffmanNode(mid_node_id, sum_frequency) # 生成父节点
    if node1.frequency >= node2.frequency: # 频率大的在左边，小的在右边
        father_node.left_child = node1
        father_node.right_child = node2
    else:
        father_node.left_child = node2
        father_node.right_child = node1
    huffman.append(father_node) # 将父节点放在节点列表中
    return father_node


# In[31]:


#node = merge_node(unmerge_node_list[0],unmerge_node_list[1])


# In[32]:


#node.frequency


# In[33]:


def build_tree(node_list): # 构建Huffman树
    while len(node_list) > 1:
        i1 = 0  # 概率最小的节点
        i2 = 1  # 概率第二小的节点
        
        # 找到概率最小和第二小的节点
        if node_list[i2].frequency < node_list[i1].frequency:
            [i1, i2] = [i2, i1]
        for i in range(2, len(node_list)):
            if node_list[i].frequency < node_list[i2].frequency:
                i2 = i
                if node_list[i2].frequency < node_list[i1].frequency:
                    [i1, i2] = [i2, i1]
        father_node = merge_node(node_list[i1], node_list[i2])  # 合并最小的两个节点
        if i1 < i2: # 删除两个节点
            node_list.pop(i2)
            node_list.pop(i1)
        elif i1 > i2:
            node_list.pop(i1)
            node_list.pop(i2)
        else:
            raise RuntimeError('i1 should not be equal to i2')
        node_list.insert(0, father_node)  # 插入新节点
    root = node_list[0] # 剩下的最后一个节点就是根节点
    return root


# In[34]:


root = build_tree(unmerge_node_list)


# In[35]:


len(huffman)


# In[36]:


def generate_huffman_code_and_path(): # 得到一个叶节点的code和path
    stack = [root]
    while len(stack) > 0:
        node = stack.pop()
        
        # 顺着左子树走
        while node.left_child or node.right_child:
            code = node.Huffman_code
            path = node.path
            node.left_child.Huffman_code = code + [1] # 往左为1
            node.right_child.Huffman_code = code + [0] # 往右为0
            node.left_child.path = path + [node.word_id] # path
            node.right_child.path = path + [node.word_id]
            stack.append(node.right_child) # 把没走过的右子树加入栈
            node = node.left_child # 一直沿着左边的走
        word_id = node.word_id 
        word_code = node.Huffman_code
        word_path = node.path
        huffman[word_id].Huffman_code = word_code
        huffman[word_id].path = word_path
        
        # 把节点计算得到的霍夫曼码、路径  写入词典的数值中
        wordid_code[word_id] = word_code
        wordid_path[word_id] = word_path
    return wordid_code, wordid_path


# In[37]:


wordid_code, wordid_path = generate_huffman_code_and_path()


# <img src="./imgs/huffman_tree.png"  width="700" height="700" align="bottom" />

# In[ ]:


wordid_code


# In[39]:


def get_all_pos_and_neg_path(): # 得到所有词作为周围词，从根节点开始走到该词的叶节点的正向节点和负向节点的id
    positive = []  # 所有词的正向路径数组
    negative = []  # 所有词的负向路径数组
    for word_id in range(len(wordid_frequency_dict)): 
        pos_id = []  # 存放一个词 路径中的正向节点id
        neg_id = []  # 存放一个词 路径中的负向节点id
        for i, code in enumerate(huffman[word_id].Huffman_code):
            if code == 1:
                pos_id.append(huffman[word_id].path[i])
            else:
                neg_id.append(huffman[word_id].path[i])
        positive.append(pos_id)
        negative.append(neg_id)
    return positive, negative


# In[40]:


positive, negative = get_all_pos_and_neg_path()


# In[41]:


positive


# In[42]:


negative


# In[ ]:




