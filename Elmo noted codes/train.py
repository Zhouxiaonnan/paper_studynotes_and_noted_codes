#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from model.data_utils import CoNLLDataset
from model.config import Config
from model.ner_model import NERModel
from model.ner_learner import NERLearner
from model.ent_model import EntModel
from model.ent_learner import EntLearner


def main():
    # create instance of config
    config = Config()
    if config.use_elmo: config.processing_word = None # 如果使用elmo，则不进行Processing_word

    #build model
    model = NERModel(config)

    # create datasets
    dev = CoNLLDataset(config.filename_dev, config.processing_word, # 验证集
                         config.processing_tag, config.max_iter, config.use_crf)
    train = CoNLLDataset(config.filename_train, config.processing_word, # 训练集
                         config.processing_tag, config.max_iter, config.use_crf)

    learn = NERLearner(config, model) # Ner学习
    learn.fit(train, dev) # 训练

if __name__ == "__main__":
    main()

