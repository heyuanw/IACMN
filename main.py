import sys
import os
import time
import tensorflow as tf
import numpy as np

import utils.reader as reader
import models.net as net
import utils.evaluation as eva
#for douban
#import utils.douban_evaluation as eva

import bin.train_and_evaluate as train
import bin.test_and_evaluate as test


os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# configure

conf = {
    "data_path": "/home/bty/why/Dialogue/DAM/data/ubuntu/data.pkl",
    "save_path": "./output/ubuntu/test/",
    "word_emb_init": "/home/bty/why/Dialogue/DAM/data/ubuntu/word_embedding.pkl",
    "init_model": None, #should be set for test
    
    "rand_seed": None, 

    "drop_dense": None,
    "drop_attention": None,

    "is_mask": True,
    "is_layer_norm": True,
    "is_positional": True,

    "dropout_keep_prob": 0.8,
    "attention_type": "bilinear",
    "dcnn_channel": 150,
    "delation_list": [1,2,4],
    "dcnn_filter_width": 3,
    "repeat_times": 2,

    "rnn_units": 200,

    "learning_rate": 1e-3,
    "vocab_size": 434513, # 434513 for ubuntu, 172131 for douban
    "emb_size": 200, # 400 for douban
    "batch_size": 100,

    "max_turn_num": 15,
    "max_turn_len": 50, 

    "max_to_keep": 5,
    "num_scan_data": 30,
    "_EOS_": 28270,    # 1 for douban data, 28270 for ubuntu
    "final_n_class": 1,

    "training": True,
}


model = net.Net(conf)
train.train(conf, model)

#test and evaluation, init_model in conf should be set
#test.test(conf, model)

