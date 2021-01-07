#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# %load E:\论文\transformer\annotated-Transformer-notebook\train1.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import numpy as np
from torch.autograd import Variable


# 生成下三角 Mask
def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
#     print(subsequent_mask)
    return torch.from_numpy(subsequent_mask) == 0

class Batch:
    "Object for holding a batch of data with mask during training."

    def __init__(self, src, trg=None, pad=0):
        # src：[batch_size, max_legth]
        self.src = src
        self.src_mask = (src != pad).unsqueeze(-2) # 非pad部分为1，pad部分为0
        if trg is not None: # 表示在训练
            
            # self.trg表示去掉每行的最后一个单词=====》相当于t-1时刻
            self.trg = trg[:, :-1] # [1,2,3,4]

            # self.trg_y表示去掉每行的第一个单词=====》相当于t时刻
            # decode 就是使用encoder和t-1时刻去预测t时刻
            self.trg_y = trg[:, 1:] # [2,3,4,5]
            self.trg_mask =                 self.make_std_mask(self.trg, pad)
            self.ntokens = (self.trg_y != pad).data.sum()

    @staticmethod
    def make_std_mask(tgt, pad):
        "Create a mask to hide padding and future words."
        tgt_mask = (tgt != pad).unsqueeze(-2)

        #    subsequent_mask(3) 生成下三角矩阵
        tgt_mask = tgt_mask & Variable(
            subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data))
        return tgt_mask

def run_epoch(data_iter, model, loss_compute):
    "Standard Training and Logging Function"
    start = time.time()
    total_tokens = 0
    total_loss = 0
    tokens = 0
    for i, batch in enumerate(data_iter):
        out = model.forward(batch.src.long(), # encoder 输入
                            batch.trg.long(), # decoder 输入
                            batch.src_mask.long(),  # encoder mask
                            batch.trg_mask.long()) # decoder mask
        # loss_compute为SimpleLossCompute方法
        # out为模型预测t时刻的单词
        # trg_y为t时刻的真实单词
        loss = loss_compute(out, batch.trg_y, batch.ntokens) # 计算 loss
        total_loss += loss # 总 loss
        total_tokens += batch.ntokens
        tokens += batch.ntokens
        if i % 50 == 1:
            elapsed = time.time() - start
            print("Epoch Step: %d Loss: %f Tokens per Sec: %f" %
                    (i, loss / batch.ntokens, tokens / elapsed))
            start = time.time()
            tokens = 0
    return total_loss / total_tokens


global max_src_in_batch, max_tgt_in_batch
def batch_size_fn(new, count, sofar):
    "Keep augmenting batch and calculate total number of tokens + padding."
    global max_src_in_batch, max_tgt_in_batch
    if count == 1:
        max_src_in_batch = 0
        max_tgt_in_batch = 0
    max_src_in_batch = max(max_src_in_batch,  len(new.src))
    max_tgt_in_batch = max(max_tgt_in_batch,  len(new.trg) + 2)
    src_elements = count * max_src_in_batch
    tgt_elements = count * max_tgt_in_batch
    return max(src_elements, tgt_elements)

# 获得数据
# V: vocab_size
# batch: batch_size
# nbatches: batch的个数
def data_gen(V, batch, nbatches):
    "Generate random data for a src-tgt copy task."
    for i in range(nbatches): # 每个batch
        data = torch.from_numpy(np.random.randint(1, V, size=(batch, 10))) # 随机生成数据
        data[:, 0] = 1
        src = Variable(data, requires_grad=False) # 用于生成 encoder 的 embedding mask
        tgt = Variable(data, requires_grad=False) # 用于生成 decoder 上三角 mask
        yield Batch(src, tgt, 0)

from smoothing import LabelSmoothing
from opt import NoamOpt
from loss import SimpleLossCompute
from model.model import make_model
from gpuloss import MultiGPULossCompute

"""
    Greedy Deocding
    测试

"""
# Train the simple copy task.
V = 11
criterion = LabelSmoothing(size=V, padding_idx=0, smoothing=0.0)
model = make_model(V, V, N=2)
model_opt = NoamOpt(model.src_embed[0].d_model, 1, 400,
        torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))

"""
for epoch in range(10):
    model.train()
    run_epoch(data_gen(V, 30, 20), model,
              SimpleLossCompute(model.generator, criterion, model_opt))
    model.eval()
    print(run_epoch(data_gen(V, 30, 5), model,
                    SimpleLossCompute(model.generator, criterion, None)))
                    
"""


# 解码
def greedy_decode(model, src, src_mask, max_len, start_symbol):
    memory = model.encode(src, src_mask) # encoder 的输出
    # 起始标记
    ys = torch.ones(1, 1).fill_(start_symbol).type_as(src.data)
    print(ys)
    for i in range(max_len-1):
        
        # decode
        out = model.decode(memory, 
                           src_mask,
                           Variable(ys),
                           Variable(subsequent_mask(ys.size(1))
                                    .type_as(src.data)))
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim = 1)
        next_word= next_word.data[0]
        ys = torch.cat([ys,
                        torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=1) # 将预测文字往后添加，并一个个进行预测
        # print("ys:"+str(ys))

    return ys

model.eval()
src = Variable(torch.LongTensor([[1,2,3,4,5,6,7,8,9,10]]) )
src_mask = Variable(torch.ones(1, 1, 10) )
# print("ys:"+str(ys))
print(greedy_decode(model, src, src_mask, max_len=10, start_symbol=1))




import dill as pickle

def prepare_dataloaders(data_pkl):

    data = pickle.load(open(data_pkl, 'rb'))

    src_vocab_size = len(data['vocab']['src'].vocab)
    trg_vocab_size = len(data['vocab']['trg'].vocab)
    trg_pad_idx = data['vocab']['trg'].vocab.stoi["<blank>"]



    fields = {'src': data['vocab']['src'], 'trg':data['vocab']['trg']}

    train = Dataset(examples=data['train'], fields=fields)
    val = Dataset(examples=data['valid'], fields=fields)


    return train, val,src_vocab_size,trg_vocab_size,trg_pad_idx


"""
    A Real World Example
"""
# For data loading.
from torchtext import data, datasets
from torchtext.data import Field, Dataset, BucketIterator

"""
if True:
    import spacy
    spacy_de = spacy.load('de')
    spacy_en = spacy.load('en')

    def tokenize_de(text):
        return [tok.text for tok in spacy_de.tokenizer(text)]

    def tokenize_en(text):
        return [tok.text for tok in spacy_en.tokenizer(text)]

    BOS_WORD = '<s>'
    EOS_WORD = '</s>'
    BLANK_WORD = "<blank>"
    SRC = data.Field(tokenize=tokenize_de, pad_token=BLANK_WORD)
    TGT = data.Field(tokenize=tokenize_en, init_token = BOS_WORD,
                     eos_token = EOS_WORD, pad_token=BLANK_WORD)

    MAX_LEN = 100
    train, val, test = datasets.IWSLT.splits(
        exts=('.de', '.en'), fields=(SRC, TGT),
        filter_pred=lambda x: len(vars(x)['src']) <= MAX_LEN and
            len(vars(x)['trg']) <= MAX_LEN)
    MIN_FREQ = 2
    SRC.build_vocab(train.src, min_freq=MIN_FREQ)
    TGT.build_vocab(train.trg, min_freq=MIN_FREQ)

"""

#先整理iterator
class MyIterator(data.Iterator):
    def create_batches(self):
        if self.train:
            def pool(d, random_shuffler):
                for p in data.batch(d, self.batch_size * 100):
                    p_batch = data.batch(
                        sorted(p, key=self.sort_key),
                        self.batch_size, self.batch_size_fn)
                    for b in random_shuffler(list(p_batch)):
                        yield b

            self.batches = pool(self.data(), self.random_shuffler)

        else:
            self.batches = []
            for b in data.batch(self.data(), self.batch_size,
                                self.batch_size_fn):
                self.batches.append(sorted(b, key=self.sort_key))


def rebatch(pad_idx, batch):
    "Fix order in torchtext to match ours"
    src, trg = batch.src.transpose(0, 1), batch.trg.transpose(0, 1)
    return Batch(src, trg, pad_idx)


datapickle="pickle路径"
train,val,src_vocab_size,trg_vocab_size,pad_idx=prepare_dataloaders(datapickle)
# GPUs to use
devices = [0, 1, 2, 3]
if True:
    # pad_idx = data['vocab']['trg'].vocab.stoi["<blank>"]
    # pad_idx = TGT.vocab.stoi["<blank>"]
    model = make_model(src_vocab_size, trg_vocab_size, N=6)
#     model.cuda()
    criterion = LabelSmoothing(size=trg_vocab_size, padding_idx=pad_idx, smoothing=0.1)
#     criterion.cuda()
    BATCH_SIZE = 12000
    train_iter = MyIterator(train, batch_size=BATCH_SIZE, device=0,
                            repeat=False, sort_key=lambda x: (len(x.src), len(x.trg)),
                            batch_size_fn=batch_size_fn, train=True)
    valid_iter = MyIterator(val, batch_size=BATCH_SIZE, device=0,
                            repeat=False, sort_key=lambda x: (len(x.src), len(x.trg)),
                            batch_size_fn=batch_size_fn, train=False)
#     model_par = nn.DataParallel(model, device_ids=devices)
None


#先下载训练好的模型文件
#!wget https://s3.amazonaws.com/opennmt-models/iwslt.pt
if False:
    model_opt = NoamOpt(model.src_embed[0].d_model, 1, 2000,
            torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))
    for epoch in range(10):
        model_par.train()
        run_epoch((rebatch(pad_idx, b) for b in train_iter),
                  model_par,
                  MultiGPULossCompute(model.generator, criterion,
                                      devices=devices, opt=model_opt))
        model_par.eval()
        loss = run_epoch((rebatch(pad_idx, b) for b in valid_iter),
                          model_par,
                          MultiGPULossCompute(model.generator, criterion,
                          devices=devices, opt=None))
        print(loss)
else:
    model = torch.load("iwslt.pt",map_location="cpu")

for i, batch in enumerate(valid_iter):
    src = batch.src.transpose(0, 1)[:1]
    src_mask = (src != SRC.vocab.stoi["<blank>"]).unsqueeze(-2)
    out = greedy_decode(model, src, src_mask,
                        max_len=60, start_symbol=TGT.vocab.stoi["<s>"])
    print("Translation:", end="\t")
    for i in range(1, out.size(1)):
        sym = TGT.vocab.itos[out[0, i]]
        if sym == "</s>": break
        print(sym, end =" ")
    print("\n翻译结束")
    print("Target:", end="\t")
    for i in range(1, batch.trg.size(0)):
        sym = TGT.vocab.itos[batch.trg.data[i, 0]]
        if sym == "</s>": break
        print(sym, end =" ")
    print()
    break


# In[ ]:


# %load E:\论文\transformer\annotated-Transformer-notebook\smoothing.py


from torch.autograd import Variable
import torch.nn as nn
import torch

# 标签平滑
class LabelSmoothing(nn.Module):
    "Implement label smoothing."
    def __init__(self, size, padding_idx, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(size_average=False) # 损失函数的形式
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing # 平滑系数
        self.size = size
        self.true_dist = None

    def forward(self, x, target):
        # x====>[batch_size*max_length-1,vocab_size]
        # target====>[batch_size*max_length-1]
        assert x.size(1) == self.size
        true_dist = x.data.clone() # 复制一份
        # fill_就是填充
        true_dist.fill_(self.smoothing / (self.size - 2))
        # scatter_修改元素
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(x, Variable(true_dist, requires_grad=False))


# In[ ]:


# %load E:\论文\transformer\annotated-Transformer-notebook\loss.py
class SimpleLossCompute:
    "A simple loss compute and train function."

    def __init__(self, generator, criterion, opt=None):
        
        self.generator = generator # model.generator，即model中的输出
        self.criterion = criterion # LabelSmoothing方法
        self.opt = opt # Noamopt

    def __call__(self, x, y, norm):
        # x对应于out，也就是预测的时刻[batch_size,max_length-1,vocab_size]
        # y对应于trg_y,也就是t时刻 [batch_size,max_length-1]
        # 通过模型得到输出
        x = self.generator(x)

        # x.contiguous().view(-1, x.size(-1)) ====>[batch_size*max_length-1,vocab_size]
        # y.contiguous().view(-1)=========>[batch_size*max_length-1]
        loss = self.criterion(x.contiguous().view(-1, x.size(-1)),
                              y.contiguous().view(-1)) / norm
        
        loss.backward() # 反向传播
        if self.opt is not None: # 优化器
            self.opt.step()
            self.opt.optimizer.zero_grad()
        return loss.data.item() * norm


# In[ ]:


# %load E:\论文\transformer\annotated-Transformer-notebook\opt.py
import torch

class NoamOpt:
    "Optim wrapper that implements rate."
    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0

    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step = None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return self.factor *                (self.model_size ** (-0.5) *
                min(step ** (-0.5), step * self.warmup ** (-1.5)))

def get_std_opt(model):
    return NoamOpt(model.src_embed[0].d_model, 2, 4000,
                   torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))


# In[ ]:




