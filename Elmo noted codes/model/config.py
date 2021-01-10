#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os


from .general_utils import get_logger
from .data_utils import get_trimmed_glove_vectors, load_vocab,         get_processing_word


class Config():
    def __init__(self, load=True):
        """Initialize hyperparameters and load vocabs

        Args:
            load_embeddings: (bool) if True, load embeddings into
                np array, else None

        """
        # directory for training outputs
        if not os.path.exists(self.dir_output):
            os.makedirs(self.dir_output)

        # create instance of logger
        self.logger = get_logger(self.path_log)

        # load if requested (default)
        # 如果load，则是从已经存在的vocab file中得到vocab
        if load:
            self.load()

    def load(self):
        """Loads vocabulary, processing functions and embeddings

        Supposes that build_data.py has been run successfully and that
        the corresponding files have been created (vocab and trimmed GloVe
        vectors)

        """
        # 1. vocabulary
        self.vocab_words = load_vocab(self.filename_words) 
        self.vocab_tags  = load_vocab(self.filename_tags) 
        self.vocab_chars = load_vocab(self.filename_chars) 

        self.nwords     = len(self.vocab_words)
        self.nchars     = len(self.vocab_chars)
        self.ntags      = len(self.vocab_tags)

        # 2. get processing functions that map str -> id
        self.processing_word = get_processing_word(self.vocab_words, # word -> id
                self.vocab_chars, lowercase=True, chars=self.use_chars)
        self.processing_tag  = get_processing_word(self.vocab_tags, # tags -> id
                lowercase=False, allow_unk=False)

        # 3. get pre-trained embeddings
        self.embeddings = (get_trimmed_glove_vectors(self.filename_trimmed)
                if self.use_pretrained else None)


    # general config
    dir_output = "results/test/"
    dir_model  = dir_output
    path_log   = dir_output + "log.txt"

    # embeddings
    dim_word = 300 # word embedding_size
    dim_char = 100 # char embeddin_size

    # glove files
    filename_glove = "data/glove.6B/glove.6B.{}d.txt".format(dim_word)
    # trimmed embeddings (created from glove_filename with build_data.py)
    filename_trimmed = "data/glove.6B.{}d.trimmed.npz".format(dim_word)
    use_pretrained = True # 是否使用预训练词向量

    # dataset
    # filename_dev = "data/coNLL/eng/eng.testa.iob"
    # filename_test = "data/coNLL/eng/eng.testb.iob"
    # filename_train = "data/coNLL/eng/eng.train.iob"

    #filename_dev = filename_test = filename_train = "data/test.txt" # test

    filename_dev = "data/valid.txt"
    filename_test = "data/test.txt"
    filename_train = "data/train.txt"

    max_iter = None # if not None, max number of examples in Dataset

    # vocab (created from dataset with build_data.py)
    filename_words = "data/words.txt"
    filename_tags = "data/tags.txt"
    filename_chars = "data/chars.txt"

    # training
    train_embeddings = False
    nepochs          = 15
    dropout          = 0.5
    batch_size       = 5
    lr_method        = "adam"
    lr               = 0.001
    lr_decay         = 0.9
    epoch_drop       = 1 # Step Decay: per # epochs to apply lr_decay
    clip             = -1 # if negative, no clipping
    nepoch_no_imprv  = 3

    # model hyperparameters
    hidden_size_char = 100 # lstm on chars
    hidden_size_lstm = 300 # lstm on word embeddings

    ner_model_path = "saves/ner_{}e_glove".format(nepochs)

    # elmo config
    use_elmo = False # 是否使用elmo
    dim_elmo = 1024 # elmo hidden state size

    # NOTE: if both chars and crf, only 1.6x slower on GPU
    use_crf = True # if crf, training is 1.7x slower on CPU
    use_chars = False if use_elmo else True#  if char embedding, training is 3.5x slower on CPU

