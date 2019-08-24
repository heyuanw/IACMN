# encoding=utf-8
import tensorflow as tf
import numpy as np
import pickle

import utils.layers as layers
import utils.operations as op

class Net(object):

    def __init__(self, conf):
        self._graph = tf.Graph()
        self._conf = conf

        if self._conf['word_emb_init'] is not None:
            print('loading word emb init')
            self._word_embedding_init = pickle.load(open(self._conf['word_emb_init'], 'rb'),encoding='iso-8859-1')
        else:
            self._word_embedding_init = None

    def build_graph(self):
        with self._graph.as_default():
            if self._conf['rand_seed'] is not None:
                rand_seed = self._conf['rand_seed']
                tf.set_random_seed(rand_seed)
                print('set tf random seed: %s' %self._conf['rand_seed'])

            #word embedding
            with tf.device('/cpu:0'), tf.name_scope("embedding"):
                self._word_embedding = tf.get_variable('word_embedding', shape=(self._conf['vocab_size'],self._conf['emb_size']), dtype=tf.float32, trainable=False)

                self.emb_placeholder = tf.placeholder(tf.float32, shape=[self._conf['vocab_size'], self._conf['emb_size']])

                self.emb_init = self._word_embedding.assign(self.emb_placeholder)


            #define placehloders
            self.turns = tf.placeholder( # context data
                tf.int32,
                shape=[None, self._conf["max_turn_num"], self._conf["max_turn_len"]])

            self.tt_turns_len = tf.placeholder( # utterance num of context
                tf.int32,
                shape=[None])

            self.every_turn_len = tf.placeholder( # length of each utterance in context
                tf.int32,
                shape=[None, self._conf["max_turn_num"]])
    
            self.response = tf.placeholder( # response data
                tf.int32, 
                shape=[None, self._conf["max_turn_len"]])

            self.response_len = tf.placeholder( # response len
                tf.int32, 
                shape=[None])

            self.label = tf.placeholder( # scale label
                tf.float32,
                shape=[None])

            self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

            #define operations
            #build response embedding
            Hr = tf.nn.embedding_lookup(self._word_embedding, self.response)
            Hr = tf.nn.dropout(Hr, self.dropout_keep_prob)

            if self._conf['is_positional']:
                with tf.variable_scope('positional'):
                    Hr = op.positional_encoding_vector(Hr, max_timescale=10)        


            with tf.variable_scope('attention_cnn_block'):
                hr_conv_list = layers.agdr_block(Hr, self._conf['repeat_times'], self._conf['delation_list'], 
                    self._conf['dcnn_filter_width'], self._conf['dcnn_channel'], self.dropout_keep_prob)

           
            list_turn_t = tf.unstack(self.turns, axis=1)
            list_turn_length = tf.unstack(self.every_turn_len, axis=1)
            
            reuse = None
            sim_turns = []
            #for every turn_t， build embedding and calculate matching vector
            for turn_t, t_turn_length in zip(list_turn_t, list_turn_length):
                
                Hu = tf.nn.embedding_lookup(self._word_embedding, turn_t)
                Hu = tf.nn.dropout(Hu, self.dropout_keep_prob)

                if self._conf['is_positional']:
                    with tf.variable_scope('positional', reuse=True):
                        Hu = op.positional_encoding_vector(Hu, max_timescale=10)

                # multi-level sim matrix of response and each utterance
                sim_matrix = [layers.Word_Sim(Hr, Hu)]

                with tf.variable_scope('attention_cnn_block', reuse=True):
                    hu_conv_list = layers.agdr_block(Hu, self._conf['repeat_times'], self._conf['delation_list'], 
                        self._conf['dcnn_filter_width'], self._conf['dcnn_channel'], self.dropout_keep_prob)
                    
                for index in range(len(hu_conv_list)):
                    with tf.variable_scope('segment_sim'):
                        sim_matrix.append(layers.Word_Sim(hr_conv_list[index], hu_conv_list[index]))

                sim_matrix = tf.stack(sim_matrix, axis=-1, name='one_matrix_stack')
                

                with tf.variable_scope('cnn_aggregation', reuse=tf.AUTO_REUSE):
                    matching_vector = layers.CNN_2d(sim_matrix, 32, 16, self.dropout_keep_prob)
                if not reuse:
                    reuse = True

                sim_turns.append(matching_vector)
            
            #aggregation with a gru
            sim = tf.stack(sim_turns, axis=1, name='matching_stack')

            with tf.variable_scope("sent_rnn"):
                sent_rnn_outputs,_ = layers.bigru_sequence(sim, 64, None, self.dropout_keep_prob) # TODO:CHECK
            
            # attention at sentence level:
            sent_atten_inputs = tf.concat(sent_rnn_outputs, 2)
            
            with tf.variable_scope("sent_atten"):
                rev_outs, alphas_sents = layers.intro_attention(sent_atten_inputs, 50)
            

            #loss and train
            with tf.variable_scope('loss'):
                self.loss, self.logits = layers.loss(rev_outs, self.label, is_clip=True)

                self.global_step = tf.Variable(0, trainable=False)
                initial_learning_rate = self._conf['learning_rate']
                self.learning_rate = tf.train.exponential_decay(
                    initial_learning_rate,
                    global_step=self.global_step,
                    decay_steps=5000, 
                    decay_rate=0.96, 
                    staircase=True)

                Optimizer = tf.train.AdamOptimizer(self.learning_rate)
                self.optimizer = Optimizer.minimize(
                    self.loss,
                    global_step=self.global_step)

                self.init = tf.global_variables_initializer()
                self.saver = tf.train.Saver(max_to_keep = self._conf["max_to_keep"])
                self.all_variables = tf.global_variables() 
                self.all_operations = self._graph.get_operations()
                self.grads_and_vars = Optimizer.compute_gradients(self.loss)

                for grad, var in self.grads_and_vars:
                    if grad is None:
                        print (var)

                self.capped_gvs = [(tf.clip_by_value(grad, -1, 1), var) for grad, var in self.grads_and_vars]
                self.g_updates = Optimizer.apply_gradients(
                    self.capped_gvs,
                    global_step=self.global_step)

            
            # summary
            grad_summaries = []
            for g, v in self.grads_and_vars:
              if g is not None:
                grad_hist_summary = tf.summary.histogram("gradient/{}/hist".format(v.name), g)
                sparsity_summary = tf.summary.scalar("gradient/{}/sparsity".format(v.name), tf.nn.zero_fraction(g))
                grad_summaries.append(grad_hist_summary)
                grad_summaries.append(sparsity_summary)

            grad_summaries_merged = tf.summary.merge(grad_summaries)

            logit_summary = tf.summary.histogram("{}".format(self.logits.name), self.logits)
            
            # Loss Summaries
            loss_summary = tf.summary.scalar("loss", self.loss)
            # Train, Dev Summaries
            self.train_summary_op = tf.summary.merge([loss_summary, logit_summary, grad_summaries_merged])
            self.dev_summary_op = tf.summary.merge([loss_summary, ])

        return self._graph

