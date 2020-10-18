#coding=utf-8
import tensorflow as tf
import pickle
import numpy as np
import Evaluate
import os
import time
import scnutils.reader_single_multi as reader
from tensorflow.python.training import moving_averages
os.environ['CUDA_VISIBLE_DEVICES']='0'
MOVING_AVERAGE_DECAY = 0.997
BN_EPSILON = 0.001
variance_scaling_initializer = tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode='FAN_IN', uniform=False)

conf = {
    "data_path": "../../data/ubuntu/data.pkl",
    "save_path": "Gcnn_v_test/version/",
    "output_path":"Gcnn_v_output/version/",
    # "word_emb_init": "./data/word_embedding.pkl",
    "init_model":"Gcnn_v_model/version/",  # should be set for test
    "embedding_file": "../../data/ubuntu/word_embedding.pkl",
    "batch_size": 32,  # 200 for test
    "epoch":5,
    "max_turn_num": 10,
    "max_turn_len": 50,
    "max_single_len":200,

    "word_layers_enc": 2,
    "word_layers_agg": 2,
    "word_layers_itg": 2,

    "filter_size":8,
    "filter_h":3,
    "_EOS_": 28270,  # 1 for douban data
    "final_n_class": 1,
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
    output=tf.divide(alph,tf.expand_dims(alph_sum+(1e-10),axis=-1))
    return output

def masked_attention_axis1(x,mask):
    '''
    :param x: [64 50 50]
    :param mask:[64 50 1]
    :return:
    '''
    beta = tf.multiply(x, mask)  # [64 50 50] * (64 50 1)  #下面是0
    beta_sum = tf.reduce_sum(beta, axis=1)#(64 50)
    output = tf.divide(beta, tf.expand_dims(beta_sum+(1e-10), axis=1))#(64 50 1)

    return output

class MyModel():
    def __init__(self,conf):
       # self._graph = tf.Graph()
        self.max_num_utterance = conf["max_turn_num"]
        self.negative_samples = 1  #负例个数可以变化

        self.is_training = False
        self.max_sentence_len = conf["max_turn_len"]
        self.max_single_len=conf["max_single_len"]
        self.word_embedding_size = 200
        self.rnn_units = 200
        self.total_words = 434513
        self.batch_size = conf['batch_size']
        self.filter_size = conf["filter_size"]
        self.filter_h = conf["filter_h"]

        def conv1d_weightnorm(inputs, layer_idx, out_dim, kernel_size, padding="SAME", dropout=1.0,
                              var_scope_name="conv_layer_", reuse=None):  # padding should take attention
            '''
            :param inputs: [64 50 200]
            :param layer_idx: 0 1 2 3
            :param out_dim: 600
            :param kernel_size: 3
            :param padding:
            :param dropout:
            :param var_scope_name:
            :param reuse:
            :return:
            '''
            with tf.variable_scope(var_scope_name, reuse=reuse):
                in_dim = int(inputs.get_shape()[-1])  # 300
                V = tf.get_variable('V', shape=[kernel_size, in_dim, out_dim], dtype=tf.float32,
                                    initializer=tf.contrib.layers.xavier_initializer(), trainable=True)
                # V [3, 300 900]
                # inputs_look=inputs #有值
                # input_conv=tf.nn.conv1d(value=inputs, filters=V, stride=1, padding=padding) # [64 50 900] #-inf
                inputs = bn(tf.nn.conv1d(value=inputs, filters=V, stride=1, padding=padding),self.is_training)  # [64 50 600]
                #  print('inputs',inputs) #
                return inputs

        def gated_linear_units(inputs, res_inputs, last_cell, layer_idx):
            '''
            :param inputs: [64 50 900]
            :param res_inputs:  [64 50 300]
            :param last_cell: [64 50 300]
            :param layer_idx: 0
            :return:
            '''
            input_shape = inputs.get_shape().as_list()
            assert len(input_shape) == 3
            dim = int(input_shape[2])  # dim=900

            # input_gate = inputs[:,:,0:dim/4]
            forget_gate = inputs[:, :, 0:dim / 3]  # (64, 50, 300)
            output_gate = inputs[:, :, dim / 3:dim * 2 / 3]  # (64, 50, 300)
            candidate = inputs[:, :, dim * 2 / 3:]  # (64, 50, 300)

            # input_gate = tf.sigmoid(input_gate)
            forget_gate = tf.sigmoid(forget_gate)
            output_gate = tf.sigmoid(output_gate)
            candidate = tf.nn.tanh(candidate)

            if layer_idx == 0:
                cell = tf.multiply(1 - forget_gate, res_inputs)  # (64, 50, 300)
            else:
                cell = tf.multiply(forget_gate, last_cell) + tf.multiply(1 - forget_gate, res_inputs)  ##(64, 50, 300)

            output = tf.multiply(output_gate, candidate) + cell
            ##tf.multiply(x,y) #x,y维度必须相等,元素对应相等
            return output, cell  # (64, 50, 300),#(64, 50, 300)

        def linear_mapping(inputs, out_dim, in_dim=None, dropout=1.0, var_scope_name="linear_mapping", reuse=None):
            '''
            :input 当[64 50 1200]
            :param out_dim:  300
            :param in_dim:
            :param dropout:
            :param var_scope_name:
            :param reuse:
            :return:[64 50 300]
            '''

            with tf.variable_scope(var_scope_name, reuse=reuse):
                input_shape = inputs.get_shape().as_list()  # static shape. may has None [64 50 1200]
                return tf.contrib.layers.fully_connected(inputs=inputs, num_outputs=out_dim, activation_fn=None,
                                                         weights_initializer=tf.random_normal_initializer
                                                         (mean=0, stddev=tf.sqrt(dropout * 1.0 / input_shape[-1])),
                                                         biases_initializer=tf.zeros_initializer())
            # 全连接成层

        def conv_encoder_stack(inputs, nhids_list, kwidths_list, dropout_dict, var_scope_name, reuse=None):
            '''
            nhids_list=[300 300 300 300] 当[300 300]
            kwidths_list=[3 3 3 3]   当[3 3 ]
            '''
            next_layer = inputs  # [64 50 300] 当[64 50 1200]
            cell = inputs  # [64 50 300]  当[64 50 1200]
            for layer_idx in range(len(nhids_list)):  # layer_idx=0 1 2 3   #layer_idx=0 1
                nout = nhids_list[layer_idx]  # nout=300 #nout的含义是输出维度是300
                if layer_idx == 0:  ##layer_idx=0 ,nin=300
                    nin = inputs.get_shape().as_list()[-1]  # nin=emb_dim 当 nin=1200
                else:
                    nin = nhids_list[
                        layer_idx - 1]  # layer_idx=1,nin=300，layer_idx=2,nin=300，layer_idx=3,nin=nhids_list[2]=300
                if nin != nout:  # 在本模型中nin=nout 此处应该是防止输入向量是200的时候的情况。当 nin=1200
                    # mapping for res add
                    res_inputs = linear_mapping(next_layer, nout, dropout=dropout_dict['src'],
                                                var_scope_name=var_scope_name + "linear_mapping_cnn_" + str(layer_idx),
                                                reuse=reuse)
                else:
                    res_inputs = next_layer  ##[64 50 300]

                next_layer = conv1d_weightnorm(inputs=next_layer, layer_idx=layer_idx, out_dim=nout * 3,
                                               kernel_size=kwidths_list[layer_idx], padding="SAME",
                                               dropout=dropout_dict['hid'],
                                               var_scope_name=var_scope_name + "conv_layer_" + str(layer_idx), reuse=reuse)
                # next_layer:NAN
                # next_layer:[64 50 900]
                next_layer, cell = gated_linear_units(next_layer, res_inputs, cell, layer_idx)
                # next_layer:(64, 50, 300),cell:(64, 50, 300)
            return next_layer  # [] #维度是多少

  #  def BuildModel(self):
       # with self._graph.as_default():
        self.utterance_ph = tf.placeholder(tf.int32, shape=(None, self.max_num_utterance, self.max_sentence_len))
        self.response_ph = tf.placeholder(tf.int32, shape=(None, self.max_sentence_len))
        self.y_true = tf.placeholder(tf.int32, shape=(None,))
        self.embedding_ph = tf.placeholder(tf.float32, shape=(self.total_words, self.word_embedding_size))#[434511,200]
        self.response_len = tf.placeholder(tf.int32, shape=(None,))
        self.all_utterance_len_ph = tf.placeholder(tf.int32, shape=(None, self.max_num_utterance))

        self.single_turn = tf.placeholder(tf.int32, shape=(None, self.max_single_len))
        self.single_turn_len = tf.placeholder(tf.int32, shape=(None,))
        self.s_r = tf.placeholder(tf.int32, shape=(None, self.max_single_len))
        self.s_r_len = tf.placeholder(tf.int32, shape=(None,))

        word_embeddings = tf.get_variable('word_embeddings_v', shape=(self.total_words,self.
                                                                      word_embedding_size), dtype=tf.float32, trainable=False) #
        self.embedding_init = word_embeddings.assign(self.embedding_ph)
        all_utterance_embeddings = tf.nn.embedding_lookup(word_embeddings, self.utterance_ph) #[batch_size 10 50 200]
        response_embeddings = tf.nn.embedding_lookup(word_embeddings, self.response_ph)  #[batch_size 50 200]
        sentence_GRU = tf.nn.rnn_cell.GRUCell(self.rnn_units, kernel_initializer=tf.orthogonal_initializer())  #self.rnn_units隐层神经元的个数
        all_utterance_embeddings = tf.unstack(all_utterance_embeddings, num=self.max_num_utterance, axis=1)
        all_utterance_len = tf.unstack(self.all_utterance_len_ph, num=self.max_num_utterance, axis=1)#max_num_utterance=10,num 为axis对应的维数

        s_turn_emb = tf.nn.embedding_lookup(word_embeddings, self.single_turn)
        s_r_emb = tf.nn.embedding_lookup(word_embeddings, self.s_r)
        A_matrix = tf.get_variable('A_matrix_v', shape=(self.rnn_units, self.rnn_units), initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
        #tf.contrib.layers.xavier_initializer()初始化权重
        final_GRU = tf.nn.rnn_cell.GRUCell(self.rnn_units, kernel_initializer=tf.orthogonal_initializer())
        reuse = None

        response_GRU_embeddings, _ = tf.nn.dynamic_rnn(sentence_GRU, response_embeddings, sequence_length=self.response_len, dtype=tf.float32,
                                                       scope='sentence_GRU')
        #response_GRU_embeddings的shape [batch_size 20 rnn_units]eg.[batch_size 50 200]
        self.response_embedding_save = response_GRU_embeddings
        response_embeddings = tf.transpose(response_embeddings, perm=[0, 2, 1]) #转置[40 200 50]
        response_GRU_embeddings = tf.transpose(response_GRU_embeddings, perm=[0, 2, 1])
        matching_vectors = []
        for utterance_embeddings, utterance_len in zip(all_utterance_embeddings, all_utterance_len):
        #utterance_embeddings 10个[40 50 200],utterance_len 10个[40]
            matrix1 = tf.matmul(utterance_embeddings, response_embeddings)
            utterance_GRU_embeddings, _ = tf.nn.dynamic_rnn(sentence_GRU, utterance_embeddings, sequence_length=utterance_len, dtype=tf.float32,
                                                            scope='sentence_GRU') #[40 50 200]
            matrix2 = tf.einsum('aij,jk->aik', utterance_GRU_embeddings, A_matrix)  # TODO:check this #A_matrix[200 200]
            matrix2 = tf.matmul(matrix2, response_GRU_embeddings)
            matrix = tf.stack([matrix1, matrix2], axis=3, name='matrix_stack') #40 50 50 2
            shape=(self.filter_h,self.filter_h)
            conv_layer = tf.layers.conv2d(matrix, filters=self.filter_size, kernel_size=shape, padding='VALID',
                                          kernel_initializer=tf.contrib.keras.initializers.he_normal(),
                                          activation=tf.nn.relu, reuse=reuse, name='conv')  # TODO: check other params
            pooling_layer = tf.layers.max_pooling2d(conv_layer, (3, 3), strides=(3, 3),
                                                    padding='VALID', name='max_pooling')  # TODO: check other params
            matching_vector = tf.layers.dense(tf.contrib.layers.flatten(pooling_layer), 100,
                                              kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                              activation=tf.tanh, reuse=reuse, name='matching_v')  # TODO: check wthether this is correct
            if not reuse:
                reuse = True
            matching_vectors.append(matching_vector) #[64 10 100]
        _, last_hidden = tf.nn.dynamic_rnn(final_GRU, tf.stack(matching_vectors, axis=0, name='matching_stack'), dtype=tf.float32,
                                           time_major=True, scope='final_GRU')  # TODO: check time_major
        #[64 50]
        s_mask_utter = length(self.single_turn)  # length: (batch_size) mask: (batch_size, max_seq_length, 1)
        s_mask_response = length(self.s_r)  # length: (64) mask: (64, 50, 1)

        dropout_dict = {'src': 1.0, 'hid': 1.0}
        conv1_utter = conv_encoder_stack(s_turn_emb, nhids_list=[100] * conf["word_layers_enc"],
                                         kwidths_list=[3] * conf["word_layers_enc"], dropout_dict=dropout_dict,
                                         var_scope_name="encoder_", reuse=None)
        conv1_response = conv_encoder_stack(s_r_emb, nhids_list=[100] * conf["word_layers_enc"],
                                            kwidths_list=[3] * conf["word_layers_enc"], dropout_dict=dropout_dict,
                                            var_scope_name="encoder_", reuse=True)

        # [batch_size, question_len, dim]
        question_output = conv1_utter
        # [batch_size, answer_len, dim]
        answer_output = conv1_response
        self.scores_unnorm = tf.matmul(question_output, answer_output, transpose_a=False, transpose_b=True)
        # tf.matmul 矩阵相乘 第一个矩阵的列数（column）等于第二个矩阵的行数（row）[64 50 50]
        self.scores_unnorm_exp = tf.exp(self.scores_unnorm)
        self.alphas = masked_attention_axis2(self.scores_unnorm_exp, tf.transpose(s_mask_response, perm=[0, 2, 1]))#(batch_size,prem_len,hyp_len)
        # self.alphas:[64 50 50]

        self.betas = masked_attention_axis1(self.scores_unnorm_exp, s_mask_utter)  # (batch_size,prem_len,hyp_len)
        # self.betas:[64 50 50]
        response_expand = tf.tile(tf.expand_dims(answer_output, 1),
                                  [1, self.max_single_len, 1, 1])  # (batch_size,prem_len,hyp_len,hidden_dim)
        # hypothesis_expand:[64 50 50 200]
        alphas = tf.expand_dims(self.alphas, -1)  # (batch_size,prem_len,hyp_len,1)
        # alphas:[64 50 50 1]
        utter_attns = tf.reduce_sum(tf.multiply(alphas, response_expand), 2)  # (batch_size,prem_len,hidden_dim)
        # premise_attns:[64 50 200]
        utter_expand = tf.tile(tf.expand_dims(question_output, 1),
                               [1, self.max_single_len, 1, 1])  # (batch_size,hyp_len,prem_len,hidden_dim)
        # premise_expand:[64 50 50 200]
        betas = tf.expand_dims(tf.transpose(self.betas, perm=[0, 2, 1]), -1)  # (batch_size,hyp_len,prem_len,1)
        # betas:[64 50 50 1]
        response_attns = tf.reduce_sum(tf.multiply(betas, utter_expand), 2)  # (batch_size,hyp_len,hidden_dim)
        # 这里应该是随意标注的注释
        ### Subcomponent Inference ###
        utter_diff = tf.abs(tf.subtract(question_output, utter_attns))  # [64 50 200]
        utter_mul = tf.multiply(question_output, utter_attns)
        response_diff = tf.abs(tf.subtract(answer_output, response_attns))
        response_mul = tf.multiply(answer_output, response_attns)

        m_a = tf.concat([question_output, utter_attns, utter_diff, utter_mul],2)  # premise_attns：[64 50 300] m_a：[None, prem_len, 4*200]
        m_b = tf.concat([answer_output, response_attns, response_diff, response_mul], 2)  # 各种维度整合的方式 #[64 50 800]

        infer_utter = conv_encoder_stack(m_a, nhids_list=[100] * conf["word_layers_agg"],
                                             kwidths_list=[3] * conf["word_layers_agg"], dropout_dict=dropout_dict,
                                             var_scope_name="inference_", reuse=None)
        infer_reaponse = conv_encoder_stack(m_b, nhids_list=[100] * conf["word_layers_agg"],
                                                kwidths_list=[3] * conf["word_layers_agg"], dropout_dict=dropout_dict,
                                                var_scope_name="inference_", reuse=True)
        question_output_2 = infer_utter
        answer_output_2 = infer_reaponse
        v1_bi = question_output_2 * s_mask_utter  # mask: (64, 200, 1) #padding 的地方归零v1_bi [64 50 300]
        v2_bi = answer_output_2 * s_mask_response

        v_1_sum = tf.reduce_sum(v1_bi, 1)  # v1_bi [batch_size prem_len hidden_dim] v_1_sum=[batch_size hidden_dim]
        v_1_ave = tf.div(v_1_sum, tf.expand_dims(tf.cast(self.single_turn_len, tf.float32)+(1e-10) ,-1))  # [batch_size  hidden_dim]

        v_2_sum = tf.reduce_sum(v2_bi, 1)
        v_2_ave = tf.div(v_2_sum, tf.expand_dims(tf.cast(self.s_r_len, tf.float32)+(1e-10) ,-1))  # [batch_size prem_len hidden_dim]

        v_1_max = tf.reduce_max(v1_bi, 1)
        v_2_max = tf.reduce_max(v2_bi, 1)

        v2 = tf.concat([v_1_ave, v_2_ave, v_1_max, v_2_max, last_hidden], 1)  # v_1_ave：[64  500]

        logits = tf.layers.dense(v2, 2, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='final_v')

        self.y_pred = tf.nn.softmax(logits)
        self.total_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.y_true, logits=logits))
        tf.summary.scalar('loss', self.total_loss)

        self.global_step = tf.Variable(0, name="global_step", trainable=False)
        self.learning_rate = tf.train.exponential_decay(0.001, global_step=self.global_step, decay_steps=1000,
                                                        decay_rate=0.96, staircase=True)
        optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
        self.train_op = optimizer.minimize(self.total_loss, global_step=self.global_step)
       # return self._graph

    def Evaluate(self,sess,val_batches,score_file_path):
        labels = []
        self.all_candidate_scores = []
        val_batch_num = len(val_batches["response"])

        eva_score_file = open(score_file_path, 'w')
        for batch_index in xrange(val_batch_num):
            feed_dict = {self.utterance_ph: np.array(val_batches["turns"][batch_index]),
                        self.all_utterance_len_ph: np.array(val_batches["every_turn_len"][batch_index]),
                        self.response_ph: np.array(val_batches["response"][batch_index]),
                        self.response_len:np.array(val_batches["response_len"][batch_index]),
                         self.single_turn: np.array(val_batches["single_turn"][batch_index]),
                         self.single_turn_len: np.array(val_batches["single_turn_len"][batch_index]),
                         self.s_r: np.array(val_batches["single_r"][batch_index]),
                         self.s_r_len: np.array(val_batches["single_r_len"][batch_index]),
                         }
            candidate_scores = sess.run(self.y_pred, feed_dict=feed_dict)
            self.all_candidate_scores.append(candidate_scores[:, 1])

            labels .extend(val_batches["label"][batch_index])
            for i in xrange(len(val_batches["label"][batch_index])):
                eva_score_file.write(str(candidate_scores[i]) +'\t'+str(val_batches["label"][batch_index][i])+ '\n')
                #labels.append(val_batches["label"][batch_index][i])
        eva_score_file.close()
        all_candidate_scores = np.concatenate(self.all_candidate_scores, axis=0)
        Evaluate.ComputeR10_1(all_candidate_scores,labels)
        Evaluate.ComputeR10_2(all_candidate_scores, labels)
        Evaluate.ComputeR10_5(all_candidate_scores, labels)
        Evaluate.ComputeR2_1(all_candidate_scores,labels)

    def TrainModel(self,conf,countinue_train = False, previous_modelpath = "model"):
        init = tf.global_variables_initializer()
        saver = tf.train.Saver()
        merged = tf.summary.merge_all()

        print('starting loading data')
        print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
        train_data, val_data, test_data = pickle.load(open(conf["data_path"], 'rb'))
        print('finish loading data')
        print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
        val_batches = reader.build_batches(test_data, conf)
        batch_num = len(train_data['y']) / conf["batch_size"]#batch_num=12500
      #  val_batch_num = len(val_batches["response"])
        print('batch_num',batch_num)
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
            for step_i in xrange(conf["epoch"]):
                print('starting shuffle train data')
                shuffle_train = reader.unison_shuffle(train_data)  # 打乱
                train_batches = reader.build_batches(shuffle_train, conf)
                print('finish building train data')

                for batch_index in range(batch_num):
                    feed_dict = { self.utterance_ph:np.array(train_batches["turns"][batch_index]),
                        self.all_utterance_len_ph: np.array(train_batches["every_turn_len"][batch_index]),
                        self.response_ph: np.array(train_batches["response"][batch_index]),
                        self.response_len:np.array(train_batches["response_len"][batch_index]),
                        self.y_true:np.array(train_batches["label"][batch_index]),
                        self.single_turn: np.array(train_batches["single_turn"][batch_index]),
                        self.single_turn_len: np.array(train_batches["single_turn_len"][batch_index]),
                        self.s_r: np.array(train_batches["single_r"][batch_index]),
                        self.s_r_len: np.array(train_batches["single_r_len"][batch_index]),
                        }

                    _, summary = sess.run([self.train_op, merged], feed_dict=feed_dict)
                    train_writer.add_summary(summary)
                    step += 1
                    if step % conf["print_step"] == 0 and step > 0 :#print_step=125 一个epoch打印100次
                        g_step, lr = sess.run([self.global_step, self.learning_rate])
                        print('epoch={i}'.format(i=step_i + 1), 'step:', step, "loss",
                              sess.run(self.total_loss, feed_dict=feed_dict),
                              "processed: [" + str(step * 1.0 / batch_num) + "]", 'gs', g_step, 'learning_rate', lr)

                    if step % conf["evaluate_step"]== 0 and step > 0:#12500的倍数就会打印
                        index = step / conf['evaluate_step']   #evaluate_file=1250
                        score_file_path = conf['save_path'] + 'score.' + str(index)
                        self.Evaluate(sess, val_batches,score_file_path)
                        print('save evaluate_step: %s' % index)
                        print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))


                if step_i > 5 and (step_i+1)%2==0: #模型保存6 8 10
                    saver.save(sess, os.path.join(conf["init_model"],"model.{0}".format(step_i+1)))
                    print(sess.run(self.total_loss, feed_dict=feed_dict))
                    print('epoch={i} save model'.format(i=step_i))
                    print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
    def TestModel(self,conf):

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
            saver.restore(sess,os.path.join(conf["init_model"],"model.4" ))
            print("sucess init %s" % conf["init_model"])

            score_file_path = conf['save_path'] + 'score.test'
            score_file = open(score_file_path, 'w')
            all_candidate_score = []
            labels=[]
            for batch_index in xrange(test_batch_num):
               # print('utterance_ph',np.array(test_batches["turns"][batch_index]).shape)
                feed_dict = {
                    self.utterance_ph:np.array( test_batches["turns"][batch_index]),
                    #_model.tt_turns_len: test_batches["tt_turns_len"][batch_index],
                    self.all_utterance_len_ph:np.array(test_batches["every_turn_len"][batch_index]),
                    self.response_ph: np.array(test_batches["response"][batch_index]),
                    self.response_len: np.array(test_batches["response_len"][batch_index]),
                   # _model.label: test_batches["label"][batch_index]
                }
                candidate_scores = sess.run(self.y_pred, feed_dict=feed_dict)
                all_candidate_score.append(candidate_scores[:, 1])
                #scores = sess.run(_model.logits, feed_dict=feed)

                for i in xrange(conf["batch_size"]):
                    score_file.write(
                        str(candidate_scores[i]) + '\t' +
                        str(test_batches["label"][batch_index][i]) + '\n')
                    labels.append(test_batches["label"][batch_index][i])
            score_file.close()

            all_candidate_scores = np.concatenate(all_candidate_score, axis=0)
            Evaluate.ComputeR10_1(all_candidate_scores, labels)
            Evaluate.ComputeR2_1(all_candidate_scores, labels)

if __name__ == "__main__":
   # scn =SCN(conf)
    #scn.BuildModel()
    #scn.TrainModel(conf)
    Gcnn = MyModel(conf)
    Gcnn.TrainModel(conf)
    #scn.TestModel(conf)
    #sess = scn.LoadModel()
    #scn.Evaluate(sess)
    #results = scn.BuildIndex(sess)
    #print(len(results))

    #scn.TrainModel()
