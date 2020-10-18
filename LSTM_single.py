# -*- coding: utf-8 -*-
from __future__ import print_function

import tensorflow as tf
import pickle
import numpy as np
import Evaluate
from tensorflow.python.training import moving_averages
import os
import time
import scnutils.readersingle as reader
import argparse
os.environ['CUDA_VISIBLE_DEVICES']='0'
MOVING_AVERAGE_DECAY = 0.997
BN_EPSILON = 0.001
variance_scaling_initializer = tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode='FAN_IN', uniform=False)

conf = {
    "data_path": "../../data/ubuntu/data.pkl",
    "save_path": "lstmsingle_test/version/",
    "output_path":"lstmsingle_output/version/",
    # "word_emb_init": "./data/word_embedding.pkl",
    "init_model": "lstmsingle_model/version/",  # should be set for test
    "embedding_file": "../../data/ubuntu/word_embedding.pkl",
    "CPU":"/cpu:0", #'/gpu:1'

    "emb_train":False,
    "word_embedding_dim":200,
    "batch_size": 20,  # 200 for test
    "epoch":5,
    "max_turn_len": 200,

    "hidden_embedding_dim":200,
    "filter_size":3,
    "filter_h":3,
    "rnn_size":200,
    "word_layers_enc":2,
    "word_layers_agg":2,
    "word_layers_itg":2,
    "_EOS_": 28270,  # 1 for douban data
    "final_n_class": 1,
    "lr":0.001
}
if not os.path.exists(conf['save_path']):
    os.makedirs(conf['save_path'])
if not os.path.exists(conf['output_path']):
    os.makedirs(conf['output_path'])
if not os.path.exists(conf['init_model']):
    os.makedirs(conf['init_model'])
def bn(x, is_training, use_bias=False):
    # Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift
    x_shape = x.get_shape()  # x_shape=[64 50 900]
    params_shape = x_shape[-1:]  # x_shape[-1:] :array([900])  x_shape[-1]:900  x_shape[:-1]:array([64 50])

    if use_bias:  # use_bias = False
        bias = tf.get_variable('bias', x_shape[-1],
                               initializer=tf.contrib.layers.xavier_initializer())  # 900
        return tf.nn.bias_add(x, bias)  # [64 50 900] 该函数要求bias是1维的，bias的维度必须和x的最后一维一样

    axis = list(range(len(x_shape) - 1))  # list(range(2))  [0 1]

    beta = tf.get_variable('beta',
                           params_shape,
                           initializer=tf.zeros_initializer())  # array([900])

    gamma = tf.get_variable('gamma',
                            params_shape,
                            initializer=tf.random_normal_initializer(mean=1.0, stddev=0.002))  # array([900])

    moving_mean = tf.get_variable('moving_mean',
                                  params_shape,
                                  initializer=tf.zeros_initializer(),
                                  trainable=False)  # array([900])

    moving_variance = tf.get_variable('moving_variance',
                                      params_shape,
                                      initializer=tf.ones_initializer(),
                                      trainable=False)  # array([900])

    # These ops will only be preformed when training.
    mean, variance = tf.nn.moments(x, axis)  # mean:[4],variance [4]求向量x的均值和方差
    update_moving_mean = moving_averages.assign_moving_average(
        moving_mean, mean, MOVING_AVERAGE_DECAY, zero_debias=False)  # if zero_debias=True, has bias
    update_moving_variance = moving_averages.assign_moving_average(
        moving_variance, variance, MOVING_AVERAGE_DECAY, zero_debias=False)  #

    def mean_var_with_update():
        with tf.control_dependencies([update_moving_mean, update_moving_variance]):
            return tf.identity(mean), tf.identity(variance)

    if is_training:  # is_training=False
        mean, var = mean_var_with_update()
        bn_x = tf.nn.batch_normalization(x, mean, var, beta, gamma, BN_EPSILON)
    else:
        bn_x = tf.nn.batch_normalization(x, moving_mean, moving_variance, beta, gamma, BN_EPSILON)

    return bn_x  # [64 50 900]
def length(x):
    """
    :param x: tensor [64 50]
    :return:  mask_prem  (64, 50, 1)
    """
    mask_prem = tf.cast(tf.cast(tf.expand_dims(x, -1), tf.bool), tf.float32)  # type=float32
    return mask_prem
def masked_attention_axis2(x,mask):
    '''
    :param x: [64 50 50]
    :param mask:[64 1 50]
    :return:[64 50 50]
    '''
    alph=tf.multiply(x,mask) #[64 50 50] * [64 1 50]
    alph_sum=tf.reduce_sum(alph,axis=2)#
    output=tf.divide(alph,tf.expand_dims(alph_sum,axis=-1))
    return output

def masked_attention_axis1(x,mask):
    '''
    :param x: [64 50 50]
    :param mask:[64 50 1]
    :return:
    '''
    beta = tf.multiply(x, mask)  # [64 50 50] * (64 50 1)  #下面是0
    beta_sum = tf.reduce_sum(beta, axis=1)#(64 50)
    output = tf.divide(beta, tf.expand_dims(beta_sum, axis=1))#(64 50 1)

    return output
def lstm_layer(inputs, input_seq_len, rnn_size, dropout_keep_prob, scope, scope_reuse=False):
    with tf.variable_scope(scope, reuse=scope_reuse) as vs:
        fw_cell = tf.contrib.rnn.LSTMCell(rnn_size, forget_bias=1.0, state_is_tuple=True, reuse=scope_reuse)
        fw_cell  = tf.contrib.rnn.DropoutWrapper(fw_cell, output_keep_prob=dropout_keep_prob)
        bw_cell = tf.contrib.rnn.LSTMCell(rnn_size, forget_bias=1.0, state_is_tuple=True, reuse=scope_reuse)
        bw_cell  = tf.contrib.rnn.DropoutWrapper(bw_cell, output_keep_prob=dropout_keep_prob)
        rnn_outputs, rnn_states = tf.nn.bidirectional_dynamic_rnn(cell_fw=fw_cell, cell_bw=bw_cell,
                                                                inputs=inputs,
                                                                sequence_length=input_seq_len,
                                                                dtype=tf.float32)
        return rnn_outputs, rnn_states
class MyModel(object):
    def __init__(self, conf):
        ## Define hyperparameters
        self.word_embedding_size = conf["word_embedding_dim"]  # 300
        emb_train = conf["emb_train"]
        self.dim = conf["hidden_embedding_dim"]  # 300
        self.max_sentence_len = conf["max_turn_len"]  # 50
        self.is_training = False
        self.total_words = 434513
        self.rnn_units = 200
      #  self.batch_size = conf['batch_size']
        self.filter_size = conf["filter_size"]
        self.filter_h = conf["filter_h"]

       # self.word_embedding_size = 200

        ## Define parameters

    ## Functions

     #   start=time.time
        self.dropout_keep_prob = tf.placeholder(tf.float32)
        self.utterance_ph = tf.placeholder(tf.int32, shape=(None,  self.max_sentence_len)) #[64 200]
        self.response_ph = tf.placeholder(tf.int32, shape=(None, self.max_sentence_len)) #[64 200]
        self.y_true = tf.placeholder(tf.int32, shape=(None,)) #[64]
        self.embedding_ph = tf.placeholder(tf.float32,shape=(self.total_words, self.word_embedding_size))  # [434511,200]
        self.response_len = tf.placeholder(tf.int32, shape=(None,)) #[64]
        self.utterance_len = tf.placeholder(tf.int32, shape=(None, ))#[64]
        word_embeddings = tf.get_variable('word_embeddings_v', shape=(self.total_words, self.word_embedding_size),
                                          dtype=tf.float32, trainable=False)
        self.embedding_init = word_embeddings.assign(self.embedding_ph)
        utterance_embeddings = tf.nn.embedding_lookup(word_embeddings, self.utterance_ph)  # [64 200 200]
        response_embeddings = tf.nn.embedding_lookup(word_embeddings, self.response_ph)  # [batch_size 50 200]
        mask_utter = length(self.utterance_ph)  # length: (batch_size) mask: (batch_size, max_seq_length, 1)
        mask_response = length(self.response_ph)  # length: (64) mask: (64, 50, 1)

        dropout_dict = {'src': 1.0, 'hid': 1.0}

        rnn_output1, rnn_states1 = lstm_layer(utterance_embeddings, self.utterance_len, conf["rnn_size"], self.dropout_keep_prob,
                                              "bidirectional_rnn", scope_reuse=None)
        rnn_output2, rnn_states2 = lstm_layer(response_embeddings, self.response_len, conf["rnn_size"], self.dropout_keep_prob,
                                              "bidirectional_rnn", scope_reuse=True)

        # [batch_size, question_len, dim]
        question_output = tf.concat(axis=2, values=rnn_output1)

        # [batch_size, answer_len, dim]
        answer_output = tf.concat(axis=2, values=rnn_output2)
        #   conv1_hyp:(64, 50, 200)
        ### Attention ###
       # self.utter_bi = conv1_utter  # (64, 200, 200)
           # print(conv1_utter.shape) #
       # self.response_bi = conv1_response
        self.scores_unnorm = tf.matmul(question_output, answer_output, transpose_a=False, transpose_b=True)
        # tf.matmul 矩阵相乘 第一个矩阵的列数（column）等于第二个矩阵的行数（row）[64 50 50]
        self.scores_unnorm_exp=tf.exp(self.scores_unnorm)
        self.alphas = masked_attention_axis2(self.scores_unnorm_exp,tf.transpose(mask_response, perm=[0, 2, 1]))  # (batch_size,prem_len,hyp_len)
        # self.alphas:[64 50 50]
        self.betas = masked_attention_axis1(self.scores_unnorm_exp, mask_utter)  # (batch_size,prem_len,hyp_len)
        # self.betas:[64 50 50]
        response_expand = tf.tile(tf.expand_dims(answer_output, 1),
                                    [1, self.max_sentence_len, 1, 1])  # (batch_size,prem_len,hyp_len,hidden_dim)
        # hypothesis_expand:[64 50 50 200]
        alphas = tf.expand_dims(self.alphas, -1)  # (batch_size,prem_len,hyp_len,1)
        # alphas:[64 50 50 1]
        utter_attns = tf.reduce_sum(tf.multiply(alphas, response_expand), 2)  # (batch_size,prem_len,hidden_dim)
        # premise_attns:[64 50 200]
        utter_expand = tf.tile(tf.expand_dims(question_output, 1),
                                 [1, self.max_sentence_len, 1, 1])  # (batch_size,hyp_len,prem_len,hidden_dim)
        # premise_expand:[64 50 50 200]
        betas = tf.expand_dims(tf.transpose(self.betas, perm=[0, 2, 1]), -1)  # (batch_size,hyp_len,prem_len,1)
        # betas:[64 50 50 1]
        response_attns = tf.reduce_sum(tf.multiply(betas, utter_expand), 2)  # (batch_size,hyp_len,hidden_dim)
        # hypothesis_attns:[64 50 200]
        print('alphas:', self.alphas.get_shape().as_list())  # [None, prem_len, hyp_len]
        print('betas:', self.betas.get_shape().as_list())  # [None, prem_len, hyp_len]
        print('premise_attns:', utter_attns.get_shape().as_list())  # [None, prem_len, 600]
        print('hypothesis_attns:', response_attns.get_shape().as_list())  # [None, hyp_len, 600]
            # 这里应该是随意标注的注释
            ### Subcomponent Inference ###
        utter_diff = tf.abs(tf.subtract(question_output, utter_attns))  # [64 50 200]
        utter_mul = tf.multiply(question_output, utter_attns)
        response_diff = tf.abs(tf.subtract(answer_output, response_attns))
        response_mul = tf.multiply(answer_output, response_attns)

        m_a = tf.concat([question_output, utter_attns, utter_diff, utter_mul],2)  # premise_attns：[64 50 300] m_a：[None, prem_len, 4*200]
        m_b = tf.concat([answer_output, response_attns, response_diff, response_mul], 2)  # 各种维度整合的方式 #[64 50 800]

        rnn_scope_layer2 = 'bidirectional_rnn_2'
        rnn_size_layer_2 = conf["rnn_size"]
        rnn_output_q_2, rnn_states_q_2 = lstm_layer(m_a, self.utterance_len, rnn_size_layer_2, self.dropout_keep_prob,
                                                    rnn_scope_layer2, scope_reuse=None)
        rnn_output_a_2, rnn_states_a_2 = lstm_layer(m_b, self.response_len, rnn_size_layer_2, self.dropout_keep_prob,
                                                    rnn_scope_layer2, scope_reuse=True)
        question_output_2 = tf.concat(axis=2, values=rnn_output_q_2)
        answer_output_2 = tf.concat(axis=2, values=rnn_output_a_2)
        v1_bi = question_output_2 * mask_utter  # mask: (64, 200, 1) #padding 的地方归零v1_bi [64 50 300]
        v2_bi = answer_output_2 * mask_response

        v_1_sum = tf.reduce_sum(v1_bi, 1)  # v1_bi [batch_size prem_len hidden_dim] v_1_sum=[batch_size hidden_dim]
        v_1_ave = tf.div(v_1_sum,tf.expand_dims(tf.cast(self.utterance_len, tf.float32)+(1e-10), -1))  # [batch_size  hidden_dim]

        v_2_sum = tf.reduce_sum(v2_bi, 1)
        v_2_ave = tf.div(v_2_sum, tf.expand_dims(tf.cast(self.response_len, tf.float32)+(1e-10),-1))  # [batch_size prem_len hidden_dim]

        v_1_max = tf.reduce_max(v1_bi, 1)
        v_2_max = tf.reduce_max(v2_bi, 1)

        v = tf.concat([v_1_ave, v_2_ave, v_1_max, v_2_max], 1)  # v_1_ave：[64  800]
        v_shape = v.get_shape().as_list()

        self.logits = tf.contrib.layers.fully_connected(inputs=v, num_outputs=2, activation_fn=None,weights_initializer=tf.contrib.layers.xavier_initializer(),biases_initializer=tf.zeros_initializer())
        #self.logits = tf.layers.dense(conv2,2,activation=tf.nn.relu,name='final_v')
        self.y_pred = tf.nn.softmax(self.logits)
        self.total_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.y_true, logits=self.logits ))
        tf.summary.scalar('loss', self.total_loss)
        self.global_step = tf.Variable(0, name="global_step", trainable=False)
        self.learning_rate = tf.train.exponential_decay(0.001, global_step=self.global_step, decay_steps=1000,
                                                        decay_rate=0.96, staircase=True)
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        self.train_op = optimizer.minimize(self.total_loss,global_step=self.global_step)


    def Evaluate(self,sess,val_batches,score_file_path):
        labels = []
        self.all_candidate_scores = []
        val_batch_num = len(val_batches["response"])

       # eva_score_file = open(score_file_path, 'w')
        for batch_index in xrange(val_batch_num):
            feed_dict = {self.utterance_ph: np.array(val_batches["turns"][batch_index]),
                        self.utterance_len: np.array(val_batches["tt_turns_len"][batch_index]),
                        self.response_ph: np.array(val_batches["response"][batch_index]),
                        self.response_len:np.array(val_batches["response_len"][batch_index]),
                        self.y_true: np.array(val_batches["label"][batch_index])
                         }
            val_loss=sess.run(self.total_loss, feed_dict=feed_dict)
          #  print('val_loss',val_loss)
            candidate_scores = sess.run(self.y_pred, feed_dict=feed_dict)
            self.all_candidate_scores.append(candidate_scores[:, 1])

            labels .extend(val_batches["label"][batch_index])
          #  for i in xrange(len(val_batches["label"][batch_index])):
             #  eva_score_file.write(str(candidate_scores[i]) +'\t'+str(val_batches["label"][batch_index][i])+ '\n')

       # eva_score_file.close()
        all_candidate_scores = np.concatenate(self.all_candidate_scores, axis=0)
        Evaluate.ComputeR10_1(all_candidate_scores,labels)
        Evaluate.ComputeR10_2(all_candidate_scores, labels)
        Evaluate.ComputeR10_5(all_candidate_scores, labels)
        Evaluate.ComputeR2_1(all_candidate_scores,labels)
    def TrainModel(self,conf,countinue_train = False, previous_modelpath = "model"):
        start=time.time()
       # conf['keep_prob'] = 0.7
        init = tf.global_variables_initializer()
        saver = tf.train.Saver()
        merged = tf.summary.merge_all()
        print('starting loading data')
        print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
        train_data, val_data, test_data = pickle.load(open(conf["data_path"], 'rb'))
        print('finish loading data')
        print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
        val_batches = reader.build_batches(test_data, conf)
        batch_num = len(train_data['y']) / conf["batch_size"]  # batch_num=12 500 15 625(64的时候)
        #  val_batch_num = len(val_batches["response"])
        print('batch_num', batch_num)
        conf["train_steps"] = conf["epoch"] * batch_num  # train_steps=2*3906
        conf["evaluate_step"] = max(1, batch_num / 1)  # max(1,1250) #每隔2500个batch保存一下
        conf["print_step"] = max(1, batch_num / 10)  # 1250    每隔100个batch打印一下
        print('configurations', conf)
        with tf.Session() as sess:
            writer = tf.summary.FileWriter(conf["output_path"], sess.graph)
            train_writer = tf.summary.FileWriter(conf["output_path"], sess.graph)

            with open(conf["embedding_file"], 'rb') as f:
                embeddings = pickle.load(f)
            if countinue_train == False:
                sess.run(init)
                sess.run(self.embedding_init, feed_dict={self.embedding_ph: embeddings})
            else:
                saver.restore(sess, previous_modelpath)

            step = 0
            learning_rate = conf['lr']
            for step_i in xrange(conf["epoch"]):
                print('starting shuffle train data')
                shuffle_train = reader.unison_shuffle(train_data)  # 打乱
                train_batches = reader.build_batches(shuffle_train, conf)
                print('finish building train data')

                for batch_index in range(batch_num):
                    feed_dict = {self.utterance_ph: np.array(train_batches["turns"][batch_index]),
                                 self.utterance_len: np.array(train_batches["tt_turns_len"][batch_index]),
                                 self.response_ph: np.array(train_batches["response"][batch_index]),
                                 self.response_len: np.array(train_batches["response_len"][batch_index]),
                                 self.y_true: np.array(train_batches["label"][batch_index]),
                                 self.dropout_keep_prob:1
                                 }
                    _, summary = sess.run([self.train_op, merged], feed_dict=feed_dict)
                    train_writer.add_summary(summary)
                    step += 1
                    if step % conf["print_step"] == 0 and step > 0:  # print_step=125 一个epoch打印100次
                        g_step, lr = sess.run([self.global_step, self.learning_rate])
                        print('epoch={i}'.format(i=step_i+1), 'step:', step, "loss",
                              sess.run(self.total_loss, feed_dict=feed_dict),
                             "processed: [" + str(step * 1.0 / batch_num) + "]",'gs',g_step,'learning_rate',lr)
                    if step % conf["evaluate_step"] == 0 and step > 0:  # 12500的倍数就会打印
                        index = step / conf['evaluate_step']  # evaluate_file=1250
                        score_file_path = conf['save_path'] + 'score.' + str(index)
                        self.Evaluate(sess, val_batches, score_file_path)
                        print('save evaluate_step: %s' % index)
                        print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))

                if (step_i + 1) >1:  # 模型保存6 8 10
                    saver.save(sess, os.path.join(conf["init_model"], "model.{0}".format(step_i + 1)))
                    print(sess.run(self.total_loss, feed_dict=feed_dict))
                    print('epoch={i} save model'.format(i=step_i+1))
                    print('learning rate', learning_rate)
                    print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
        end = time.time()
        gap = (end - start) / 3600
        print('train time:%.4f h' % gap)
    def TestModel(self, conf):
        start=time.time()
        conf['keep_prob'] = 1
        if not os.path.exists(conf['save_path']):
            os.makedirs(conf['save_path'])
        print('beging test starting loading data')
        print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
        train_data, val_data, test_data = pickle.load(open(conf["data_path"], 'rb'))
        print('finish loading data')

        test_batches = reader.build_batches(test_data, conf)

        print("finish building test batches")
        print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))

        # refine conf
        test_batch_num = len(test_batches["response"])

        with tf.Session() as sess:
            saver = tf.train.Saver()
            # with tf.Session() as sess:
            # sess.run(init)
            saver.restore(sess, os.path.join(conf["init_model"], "model.3"))
            print("sucess init %s" % conf["init_model"])

            score_file_path = conf['save_path'] + 'score.test'
            score_file = open(score_file_path, 'w')
            all_candidate_score = []
            labels = []
            for batch_index in xrange(test_batch_num):
                # print('utterance_ph',np.array(test_batches["turns"][batch_index]).shape)
                feed_dict = {
                    self.utterance_ph: np.array(test_batches["turns"][batch_index]),
                    # _model.tt_turns_len: test_batches["tt_turns_len"][batch_index],
                    self.all_utterance_len_ph: np.array(test_batches["every_turn_len"][batch_index]),
                    self.response_ph: np.array(test_batches["response"][batch_index]),
                    self.response_len: np.array(test_batches["response_len"][batch_index]),
                    # _model.label: test_batches["label"][batch_index]
                }
              #  last_hidden = sess.run(self.last_hidden, feed_dict=feed_dict)
               # print('last_hidden', last_hidden.shape)
                candidate_scores = sess.run(self.y_pred, feed_dict=feed_dict)
                all_candidate_score.append(candidate_scores[:, 1])
                # scores = sess.run(_model.logits, feed_dict=feed)

                for i in xrange(conf["batch_size"]):
                    score_file.write(
                        str(candidate_scores[i]) + '\t' +
                        str(test_batches["label"][batch_index][i]) + '\n')
                    labels.append(test_batches["label"][batch_index][i])
            score_file.close()

            all_candidate_scores = np.concatenate(all_candidate_score, axis=0)
            Evaluate.ComputeR10_1(all_candidate_scores, labels)
            Evaluate.ComputeR10_2(all_candidate_scores, labels)
            Evaluate.ComputeR10_5(all_candidate_scores, labels)
            Evaluate.ComputeR2_1(all_candidate_scores, labels)
            #douban_evaluation.evaluate(all_candidate_scores, labels)

        end = time.time()
        gap = (end - start) / 3600
        print('test time:%.4f h' % gap)
if __name__ == "__main__":
    Gcnn = MyModel(conf)
    Gcnn.TrainModel(conf)
   # Gcnn.TestModel(conf)
    '''
    ### Inference Composition ###
    # infer_prem=[batch_size prem_len hidden_dim] ?  [batch_size prem_len 2*hidden_dim]
    infer_utter = conv_encoder_stack(m_a, nhids_list=[200] * conf["word_layers_agg"],
                                    kwidths_list=[3] * conf["word_layers_agg"], dropout_dict=dropout_dict,
                                    var_scope_name="inference_", reuse=None)
    infer_reaponse = conv_encoder_stack(m_b, nhids_list=[200] * conf["word_layers_agg"],
                                   kwidths_list=[3] * conf["word_layers_agg"], dropout_dict=dropout_dict,
                                   var_scope_name="inference_", reuse=True)
    # infer_prem:[64 50 300]
    ### Pooling Layer ###
    v1_bi = infer_utter * mask_utter  # mask: (64, 50, 1) #padding 的地方归零v1_bi [64 50 300]
    v2_bi = infer_reaponse * mask_response

    v_1_sum = tf.reduce_sum(v1_bi, 1)  # v1_bi [batch_size prem_len hidden_dim] v_1_sum=[batch_size hidden_dim]
    v_1_ave = tf.div(v_1_sum, tf.expand_dims(tf.cast(utterance_len, tf.float32), -1))  # [batch_size  hidden_dim]

    v_2_sum = tf.reduce_sum(v2_bi, 1)
    v_2_ave = tf.div(v_2_sum, tf.expand_dims(tf.cast(self.response_len, tf.float32), -1))  # [batch_size prem_len hidden_dim]

    v_1_max = tf.reduce_max(v1_bi, 1)
    v_2_max = tf.reduce_max(v2_bi, 1)

    v = tf.concat([v_1_ave, v_2_ave, v_1_max, v_2_max], 1)  # v_1_ave：[batch_size  hidden_dim]
    # v [batch_size  hidden_dim*4]

    # MLP layer
    v_shape = v.get_shape().as_list()
    self.W_mlp = tf.Variable(tf.random_normal([v_shape[-1], self.dim], stddev=0.1))
    self.b_mlp = tf.Variable(tf.random_normal([self.dim], stddev=0.1))

    self.W_cl = tf.Variable(tf.random_normal([self.dim, 2], stddev=0.1))
    self.b_cl = tf.Variable(tf.random_normal([2], stddev=0.1))

    h_mlp = tf.nn.relu(tf.matmul(v, self.W_mlp) + self.b_mlp)

    # Dropout applied to classifier
    h_drop = tf.nn.dropout(h_mlp, self.keep_rate_ph)

    # Get prediction
    self.logits = tf.matmul(h_drop, self.W_cl) + self.b_cl

    # Define the cost function
    self.total_cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.y, logits=self.logits))
    print('self.total_cost:', self.total_cost.get_shape().as_list())
    for ele in tf.global_variables():
        print(ele.op.name)
    '''
