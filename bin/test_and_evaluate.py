import sys
import os
import time

import pickle
import tensorflow as tf
import numpy as np

import utils.reader as reader
import utils.evaluation as eva
#import utils.douban_evaluation as eva

def test(conf, _model):
    
    if not os.path.exists(conf['save_path']):
        os.makedirs(conf['save_path'])

    # load data
    print('starting loading data')
    print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))
    train_data, val_data, test_data = pickle.load(open(conf["data_path"], 'rb'))
    
    print('finish loading data')
    print('train data len:', len(train_data['y']))
    print('val data len:', len(val_data['y']))
    print('test data len:', len(test_data['y']))

    test_batches = reader.build_batches('test',test_data, conf)

    print("finish building test batches")
    print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))

    # refine conf
    test_batch_num = len(test_batches["response"])
    print('test_batch_num:',test_batch_num)

    print('configurations: %s' %conf)


    _graph = _model.build_graph()
    print('build graph sucess')
    print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))

    with tf.Session(graph=_graph) as sess:
        #_model.init.run();
        _model.saver.restore(sess, conf["init_model"])
        print("sucess init %s" %conf["init_model"])

        batch_index = 0
        step = 0

        score_file_path = conf['save_path'] + 'score.test'
        score_file = open(score_file_path, 'w')

        print('starting test')
        print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))
        for batch_index in range(test_batch_num):
                
            feed = { 
                _model.turns: test_batches["turns"][batch_index],
                _model.tt_turns_len: test_batches["tt_turns_len"][batch_index],
                _model.every_turn_len: test_batches["every_turn_len"][batch_index],
                _model.response: test_batches["response"][batch_index],
                _model.response_len: test_batches["response_len"][batch_index],
                _model.label: test_batches["label"][batch_index],
                _model.dropout_keep_prob: 1.0
                }
                
            scores = sess.run(_model.logits, feed_dict = feed)
            
            for i in range(len(scores)):
                score_file.write(
                    str(scores[i]) + '\t' + 
                    str(test_batches["label"][batch_index][i]) + '\n') # the ith sample in the batch_index batch


        score_file.close()
        print('finish test')
        print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))

        
        #write evaluation result
        result = eva.evaluate(score_file_path)
        result_file_path = conf["save_path"] + "result.test"
        with open(result_file_path, 'w') as out_file:
            for p_at in result:
                out_file.write(str(p_at) + '\n')
        print('finish evaluation')
        print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))
        

                    
