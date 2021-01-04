#!/usr/bin/env python
# coding: utf-8

# In[2]:


import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim

torch.manual_seed(1)


# In[3]:


def argmax(vec):
    # return the argmax as a python int
    # 返回vec的dim为1维度上的最大值索引
    _, idx = torch.max(vec, 1)
    return idx.item()

def prepare_sequence(seq, to_ix):
    # 将句子转化为ID
    idxs = [to_ix[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)

# Compute log sum exp in a numerically stable way for the forward algorithm
# 前向算法是不断累积之前的结果，这样就会有个缺点
# 指数和累积到一定程度后，会超过计算机浮点值的最大值，变成inf，这样取log后也是inf
# 为了避免这种情况，用一个合适的值clip去提指数和的公因子，这样就不会使某项变得过大而无法计算
# SUM = log(exp(s1)+exp(s2)+...+exp(s100))
#     = log{exp(clip)*[exp(s1-clip)+exp(s2-clip)+...+exp(s100-clip)]}
#     = clip + log[exp(s1-clip)+exp(s2-clip)+...+exp(s100-clip)]
# where clip=max
def log_sum_exp(vec):
    max_score = vec[0, argmax(vec)] # 得到最大值
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1]) # 扩展维度
    return max_score + torch.log(torch.sum(torch.exp(vec - max_score_broadcast))) # 计算分数


# In[4]:


class BiLSTM_CRF(nn.Module):

    def __init__(self, vocab_size, tag_to_ix, embedding_dim, hidden_dim):
        super(BiLSTM_CRF, self).__init__()
        self.embedding_dim = embedding_dim    # word embedding dim
        self.hidden_dim = hidden_dim          # Bi-LSTM hidden dim
        self.vocab_size = vocab_size          # 词表大小
        self.tag_to_ix = tag_to_ix            # 标签2id BEIOS -> id
        self.tagset_size = len(tag_to_ix)     # 标签的数量

        self.word_embeds = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2, # 因为之后要进行正反方向的concat
                            num_layers = 1, bidirectional = True)

        # Maps the output of the LSTM into tag space.
        # 将BiLSTM提取的特征向量映射到特征空间，即经过全连接得到发射分数
        self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size)

        # Matrix of transition parameters.  Entry i,j is the score of transitioning *to* i *from* j.
        # 转移矩阵的参数初始化，transitions[i,j]代表的是从第j个tag转移到第i个tag的转移分数
        # torch.randn：从标准正态分布中产生随机数 tagset_size * tagset_size
        self.transitions = nn.Parameter(
            torch.randn(self.tagset_size, self.tagset_size))

        # These two statements enforce the constraint that we never transfer
        # to the start tag and we never transfer from the stop tag
        # 初始化所有其他tag转移到START_TAG的分数非常小，即不可能由其他tag转移到START_TAG
        # 初始化STOP_TAG转移到所有其他tag的分数非常小，即不可能由STOP_TAG转移到其他tag
        self.transitions.data[tag_to_ix[START_TAG], :] = -10000
        self.transitions.data[:, tag_to_ix[STOP_TAG]] = -10000

        self.hidden = self.init_hidden()
        
    # 初始化LSTM的参数
    def init_hidden(self):
        
        # torch.randn：从标准正态分布中产生随机数 2 * 1 * (hidden_dim // 2)
        # 正向 hidden state + cell state，反向 hidden state + cell state
        return (torch.randn(2, 1, self.hidden_dim // 2), 
                torch.randn(2, 1, self.hidden_dim // 2))
    
    # 初始化LSTM的参数
    def _get_lstm_features(self, sentence):
        
        self.hidden = self.init_hidden() # 初始化 hidden_state
        
        # 维度转换（lstm必须输入三维数据），sentence_length * 1 * embedding_size
        embeds = self.word_embeds(sentence).view(len(sentence), 1, -1) 
        
#         pytorch: sentence_length * batch_size * embedding_size
#         tensorflow: batch_size * sentence_size * embedding_size
        
        # lstm_out：每个时间步的输出，self.hidden：最后一个时间步的输出
        # lstm_out：sentence_length * 1 * hidden_dim（双向LSTM，正向和反向的hidden state进行concat）
        lstm_out, self.hidden = self.lstm(embeds, self.hidden) 
        lstm_out = lstm_out.view(len(sentence), self.hidden_dim) # 维度转换（全连接层必须输入二维数据） sentence_length * hidden_dim
        lstm_feats = self.hidden2tag(lstm_out) # 全连接层 sentence_length * tagset_size
        return lstm_feats # 发射分数，每个单词都有 tagset_size 个发射分数
    
    # 计算一条路径的分数，用来计算最优路径的分数
    def _score_sentence(self, feats, tags):
        
        # Gives the score of a provided tag sequence
        # 计算给定tag序列的分数，即一条路径的分数
        score = torch.zeros(1)
        tags = torch.cat([torch.tensor([self.tag_to_ix[START_TAG]], dtype=torch.long), tags]) # 在tags开头接上start的id
        for i, feat in enumerate(feats): # 每个时间步，每个时间步对应 tagset_size 个发射分数
            
            # 递推计算路径分数：转移分数 + 发射分数
            # 从时间步 i 转移到时间步 i+1 的转移分数 + i+1 时间步的发射分数
            # tags[0]是start，而feats不是从start开始的，所以这里发射分数是用feat[tags[i+1]]
            score = score + # 之前的score
                    self.transitions[tags[i + 1], tags[i]] + # 当前步的转移score
                    feat[tags[i + 1]] # 当前步的发射score
        score = score + self.transitions[self.tag_to_ix[STOP_TAG], tags[-1]] # 最后加上转移到stop的分数
        return score # 一条路径的分数
    
    # Do the forward algorithm to compute the partition function
    # 通过前向算法递推计算所有可能路径的分数之和
    def _forward_alg(self, feats):
        
        # 初始化 1 * tagset_size
        init_alphas = torch.full((1, self.tagset_size), -10000.)
        
        # START_TAG has all of the score.
        # 初始化step 0即START位置的发射分数，START_TAG取0其他位置取-10000
        init_alphas[0][self.tag_to_ix[START_TAG]] = 0.

        # Wrap in a variable so that we will get automatic backprop
        # 将初始化START位置为0的发射分数赋值给previous
        previous = init_alphas

        # Iterate through the sentence
        # 迭代整个句子，obs：当前步的发射分数矩阵
        for obs in feats:
            
            # The forward tensors at this timestep
            # 当前时间步的前向tensor
            alphas_t = []
            for next_tag in range(self.tagset_size):
                
                # broadcast the emission score: it is the same regardless of the previous tag
                # 取出当前tag的发射分数 1 * tagset_size
                # 比如有三个tag，且next_tag为tag1，emit_score就是当前步对应文字为tag1的发射分数
                # 实际上为1个分数，因此要扩展到tagset_size维度
                emit_score = obs[next_tag].view(1, -1).expand(1, self.tagset_size)
                
                # the ith entry of trans_score is the score of transitioning to next_tag from i
                # 取出当前tag由之前tag转移过来的转移分数 1 * tagset_size
                # 即由tag1, tag2, tag3 转移到 tag1 的转移分数
                trans_score = self.transitions[next_tag].view(1, -1)
                
                # The ith entry of next_tag_var is the value for the edge (i -> next_tag) before we do log-sum-exp
                # 当前路径的分数：之前时间步分数 + 转移分数 + 发射分数 1 * tagset_size
                next_tag_var = previous + trans_score + emit_score
                
                # The forward variable for this tag is log-sum-exp of all the scores.
                # 对当前分数取log-sum-exp
                # 对于一开始的 start，因为只有从start开始走，所以其他节点设为-10000，在exp的过程中得到0
                # 对所有的tag进行计算后，得到的alphas_t的形状是 1 * tagset_size，表示当前步的score
                alphas_t.append(log_sum_exp(next_tag_var).view(1))
                
            # 更新 previous 递推计算下一个时间步
            # torch.cat：拼接
            previous = torch.cat(alphas_t).view(1, -1)
            
        # 考虑最终转移到STOP_TAG
        terminal_var = previous + self.transitions[self.tag_to_ix[STOP_TAG]]
        
        # 计算最终的分数（所有路径的分数之和）
        scores = log_sum_exp(terminal_var)
        return scores

    # 通过维特比算法得到最优路径和最优路径的分数
    def _viterbi_decode(self, feats):
        backpointers = []

        # Initialize the viterbi variables in log space
        # 初始化viterbi的previous变量
        init_vvars = torch.full((1, self.tagset_size), -10000.)
        init_vvars[0][self.tag_to_ix[START_TAG]] = 0
        previous = init_vvars
        
        # obs：当前步的发射分数矩阵
        for obs in feats:
            
            # holds the backpointers for this step
            # 保存当前时间步的回溯指针
            bptrs_t = []
            
            # holds the viterbi variables for this step
            # 保存当前时间步的viterbi变量
            viterbivars_t = []  

            for next_tag in range(self.tagset_size):
                # next_tag_var[i] holds the viterbi variable for tag i at the
                # previous step, plus the score of transitioning
                # from tag i to next_tag.
                # We don't include the emission scores here because the max
                # does not depend on them (we add them in below)
                # 维特比算法记录最优路径时只考虑上一步的分数以及上一步tag转移到当前tag的转移分数
                # 并不取决与当前tag的发射分数，因为对于同一个next_tag，每个路径都要乘以相同的发射分数，对于对比路径来说没有作用
                # 如果有三个tag，这里算的是前一步的三个tag到达下一步的一个tag的分数
                # 即tag1,tag2,tag3到达tag1的分数，在内层循环中再计算三个tag到tag2的分数，以及三个tag到tag3的分数
                next_tag_var = previous + self.transitions[next_tag] # 三个分数
                best_tag_id = argmax(next_tag_var) # 取最大分数对应的index
                bptrs_t.append(best_tag_id) # 到达该tag的最优路径的上一个tag
                viterbivars_t.append(next_tag_var[0][best_tag_id].view(1)) # 最高得分数
                
            # Now add in the emission scores, and assign forward_var to the set
            # of viterbi variables we just computed
            # 更新previous，加上当前tag的发射分数obs
            previous = (torch.cat(viterbivars_t) + obs).view(1, -1)
            
            # 回溯指针记录当前时间步各个tag来源前一步的tag
            backpointers.append(bptrs_t)
        
        # 考虑转移到STOP_TAG的转移分数
        terminal_var = previous + self.transitions[self.tag_to_ix[STOP_TAG]]
        best_tag_id = argmax(terminal_var)
        path_score = terminal_var[0][best_tag_id]
        
        # 双层循环结束，每一步都计算完成
        # 通过回溯指针解码出最优路径
        best_path = [best_tag_id]
        
        # best_tag_id作为线头，反向遍历backpointers找到最优路径
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)
            
        # Pop off the start tag (we dont want to return that to the caller)
        # 去除START_TAG
        start = best_path.pop()
        assert start == self.tag_to_ix[START_TAG]  # Sanity check
        best_path.reverse()
        return path_score, best_path

    def neg_log_likelihood(self, sentence, tags):
        # CRF损失函数由两部分组成，真实路径的分数和所有路径的总分数。
        # 真实路径的分数应该是所有路径中分数最高的。
        # log真实路径的分数/log所有可能路径的分数，越大越好，构造crf loss函数取反，loss越小越好
        feats = self._get_lstm_features(sentence)
        forward_score = self._forward_alg(feats)
        gold_score = self._score_sentence(feats, tags)
        return forward_score - gold_score

    def forward(self, sentence):  # dont confuse this with _forward_alg above.
        # Get the emission scores from the BiLSTM
        # 通过BiLSTM提取发射分数
        lstm_feats = self._get_lstm_features(sentence)

        # Find the best path, given the features.
        # 根据发射分数以及转移分数，通过viterbi解码找到一条最优路径
        score, tag_seq = self._viterbi_decode(lstm_feats)
        return score, tag_seq


# In[5]:


START_TAG = "<START>"
STOP_TAG = "<STOP>"
EMBEDDING_DIM = 5
HIDDEN_DIM = 4

# Make up some training data
# 构造一些训练数据
training_data = [(
    "the wall street journal reported today that apple corporation made money".split(),
    "B I I I O O O B I O O".split()
), (
    "georgia tech is a university in georgia".split(),
    "B I O O O O B".split()
)]

word_to_ix = {}
for sentence, tags in training_data:
    for word in sentence:
        if word not in word_to_ix:
            word_to_ix[word] = len(word_to_ix) # 字2id表

tag_to_ix = {"B": 0, "I": 1, "O": 2, START_TAG: 3, STOP_TAG: 4}

model = BiLSTM_CRF(len(word_to_ix), tag_to_ix, EMBEDDING_DIM, HIDDEN_DIM)
optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-4)

# Check predictions before training
# 训练前检查模型预测结果
with torch.no_grad():
    precheck_sent = prepare_sequence(training_data[0][0], word_to_ix)
    precheck_tags = torch.tensor([tag_to_ix[t] for t in training_data[0][1]], dtype=torch.long)
    print(model(precheck_sent))

# Make sure prepare_sequence from earlier in the LSTM section is loaded
for epoch in range(300):  # again, normally you would NOT do 300 epochs, it is toy data
    for sentence, tags in training_data:
        # Step 1. Remember that Pytorch accumulates gradients.
        # We need to clear them out before each instance
        # 第一步，pytorch梯度累积，需要清零梯度
        model.zero_grad()

        # Step 2. Get our inputs ready for the network, that is,
        # turn them into Tensors of word indices.
        # 第二步，将输入转化为tensors
        sentence_in = prepare_sequence(sentence, word_to_ix)
        targets = torch.tensor([tag_to_ix[t] for t in tags], dtype=torch.long)

        # Step 3. Run our forward pass.
        # 进行前向计算，取出crf loss
        loss = model.neg_log_likelihood(sentence_in, targets)

        # Step 4. Compute the loss, gradients, and update the parameters by
        # calling optimizer.step()
        # 第四步，计算loss，梯度，通过optimier更新参数
        loss.backward()
        optimizer.step()

# Check predictions after training
# 训练结束查看模型预测结果，对比观察模型是否学到
with torch.no_grad():
    precheck_sent = prepare_sequence(training_data[0][0], word_to_ix)
    print(model(precheck_sent))
# We got it!

