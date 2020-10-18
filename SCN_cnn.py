#coding=utf-8
import tensorflow as tf
import pickle
import utils
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import Evaluate
import os
import time
import scnutils.reader as reader
import argparse
'''
第一步修改SCN_cnn.py参数"max_turn_num": 9
nohup python -u SCN_cnn.py  >log/9SCN.log 2>&1 & 
'''
os.environ['CUDA_VISIBLE_DEVICES']='0'
parser = argparse.ArgumentParser()
parser.add_argument('--turn', type=int)
args=parser.parse_args()
conf = {
    "max_turn_num": int(args.turn),
    "data_path": "../../data/ubuntu/data.pkl",
    "save_path": "cnn_test/version/",
    "output_path":"output/version/",
    # "word_emb_init": "./data/word_embedding.pkl",
    "init_model": "model/version/",  # should be set for test
    "embedding_file": "../../data/ubuntu/word_embedding.pkl",
    "batch_size": 64,  # 200 for test
    "epoch":5,
    "max_turn_len": 50,
    "filter_size":8,
    "filter_h":3,
    "_EOS_": 28270,  # 1 for douban data
    "final_n_class": 1,
}
print("turns:",conf["max_turn_num"])
if not os.path.exists(conf['save_path']):
    os.makedirs(conf['save_path'])
if not os.path.exists(conf['output_path']):
    os.makedirs(conf['output_path'])
if not os.path.exists(conf['init_model']):
    os.makedirs(conf['init_model'])

class SCN():
    def __init__(self,conf):
       # self._graph = tf.Graph()
        self.max_num_utterance = conf["max_turn_num"]
        self.negative_samples = 1  #负例个数可以变化
        self.max_sentence_len = conf["max_turn_len"]
        self.word_embedding_size = 200
        self.rnn_units = 200
        self.total_words = 434513
        self.batch_size = conf['batch_size']
        self.filter_size = conf["filter_size"]
        self.filter_h = conf["filter_h"]

    def LoadModel(self):
        #init = tf.global_variables_initializer()
        saver = tf.train.Saver()
        sess = tf.Session()
        #with tf.Session() as sess:
            #sess.run(init)
        saver.restore(sess,"neg5model\\model.5")
        return sess
        # Later, launch the model, use the saver to restore variables from disk, and
        # do some work with the model.
        # with tf.Session() as sess:
        #     # Restore variables from disk.
        #     saver.restore(sess, "/model/model.5")
        #     print("Model restored.")

    def BuildModel(self):
       # with self._graph.as_default():
        self.utterance_ph = tf.placeholder(tf.int32, shape=(None, self.max_num_utterance, self.max_sentence_len))
        self.response_ph = tf.placeholder(tf.int32, shape=(None, self.max_sentence_len))
        self.y_true = tf.placeholder(tf.int32, shape=(None,))
        self.embedding_ph = tf.placeholder(tf.float32, shape=(self.total_words, self.word_embedding_size))#[434511,200]
        self.response_len = tf.placeholder(tf.int32, shape=(None,))
        self.all_utterance_len_ph = tf.placeholder(tf.int32, shape=(None, self.max_num_utterance))
        word_embeddings = tf.get_variable('word_embeddings_v', shape=(self.total_words,self.
                                                                      word_embedding_size), dtype=tf.float32, trainable=False) #
        self.embedding_init = word_embeddings.assign(self.embedding_ph)
        all_utterance_embeddings = tf.nn.embedding_lookup(word_embeddings, self.utterance_ph) #[batch_size 10 50 200]
        response_embeddings = tf.nn.embedding_lookup(word_embeddings, self.response_ph)  #[batch_size 50 200]
        sentence_GRU = tf.nn.rnn_cell.GRUCell(self.rnn_units, kernel_initializer=tf.orthogonal_initializer())  #self.rnn_units隐层神经元的个数
        all_utterance_embeddings = tf.unstack(all_utterance_embeddings, num=self.max_num_utterance, axis=1)
        all_utterance_len = tf.unstack(self.all_utterance_len_ph, num=self.max_num_utterance, axis=1)#max_num_utterance=10,num 为axis对应的维数
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
            matrix = tf.stack([matrix1, matrix2], axis=3, name='matrix_stack')
            shape=(self.filter_h,self.filter_h)
            conv_layer = tf.layers.conv2d(matrix, filters=self.filter_size, kernel_size=shape, padding='VALID',
                                          kernel_initializer=tf.contrib.keras.initializers.he_normal(),
                                          activation=tf.nn.relu, reuse=reuse, name='conv')  # TODO: check other params
            pooling_layer = tf.layers.max_pooling2d(conv_layer, (3, 3), strides=(3, 3),
                                                    padding='VALID', name='max_pooling')  # TODO: check other params
            matching_vector = tf.layers.dense(tf.contrib.layers.flatten(pooling_layer), 50,
                                              kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                              activation=tf.tanh, reuse=reuse, name='matching_v')  # TODO: check wthether this is correct
            if not reuse:
                reuse = True
            matching_vectors.append(matching_vector) #[64 10 50]
        _, last_hidden = tf.nn.dynamic_rnn(final_GRU, tf.stack(matching_vectors, axis=0, name='matching_stack'), dtype=tf.float32,
                                           time_major=True, scope='final_GRU')  # TODO: check time_major
        logits = tf.layers.dense(last_hidden, 2, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='final_v')
        self.y_pred = tf.nn.softmax(logits)
        self.total_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.y_true, logits=logits))
        tf.summary.scalar('loss', self.total_loss)
        optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
        self.train_op = optimizer.minimize(self.total_loss)
       # return self._graph

    def Evaluate(self,sess,val_batches,score_file_path):
        labels = []
        self.all_candidate_scores = []
        val_batch_num = len(val_batches["response"])

      #  eva_score_file = open(score_file_path, 'w')
        for batch_index in xrange(val_batch_num):
            feed_dict = {self.utterance_ph: np.array(val_batches["turns"][batch_index]),
                        self.all_utterance_len_ph: np.array(val_batches["every_turn_len"][batch_index]),
                        self.response_ph: np.array(val_batches["response"][batch_index]),
                        self.response_len:np.array(val_batches["response_len"][batch_index]),
                         }
            candidate_scores = sess.run(self.y_pred, feed_dict=feed_dict)
            self.all_candidate_scores.append(candidate_scores[:, 1])

            labels .extend(val_batches["label"][batch_index])
           # for i in xrange(len(val_batches["label"][batch_index])):
            #    eva_score_file.write(str(candidate_scores[i]) +'\t'+str(val_batches["label"][batch_index][i])+ '\n')
                #labels.append(val_batches["label"][batch_index][i])
       # eva_score_file.close()
        all_candidate_scores = np.concatenate(self.all_candidate_scores, axis=0)
        Evaluate.ComputeR10_1(all_candidate_scores,labels)
        Evaluate.ComputeR10_2(all_candidate_scores, labels)
        Evaluate.ComputeR10_5(all_candidate_scores, labels)
        Evaluate.ComputeR2_1(all_candidate_scores,labels)
       # eva_score_file.close()
    def TrainModel(self,conf,countinue_train = False, previous_modelpath = "model"):
        init = tf.global_variables_initializer()
        saver = tf.train.Saver()
        merged = tf.summary.merge_all()

        print('starting loading data')
        print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
        train_data, val_data, test_data = pickle.load(open(conf["data_path"], 'rb'))
        print('finish loading data')
        print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
        val_batches = reader.build_batches(val_data, conf)
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
                        self.y_true:np.array(train_batches["label"][batch_index])
                        }

                    _, summary = sess.run([self.train_op, merged], feed_dict=feed_dict)
                    train_writer.add_summary(summary)
                    step += 1
                    if step % conf["print_step"] == 0 and step > 0 :#print_step=125 一个epoch打印100次
                        print('epoch={i}'.format(i=step_i),'step:',step,"loss",sess.run(self.total_loss, feed_dict=feed_dict),"processed: [" + str(step * 1.0 / batch_num) + "]")

                    if step % conf["evaluate_step"]== 0 and step > 0:#12500的倍数就会打印
                        index = step / conf['evaluate_step']   #evaluate_file=1250
                        score_file_path = conf['save_path'] +'%d_turns_score'%(conf['max_turn_num'])
                        self.Evaluate(sess, val_batches,score_file_path)
                        print('save evaluate_step: %s' % index)
                        print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))

                if step_i +1> 0 : #模型保存6 8 10
                   # saver.save(sess, os.path.join(conf["init_model"],"model.{0}".format(step_i+1)))
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
    scn =SCN(conf)
    scn.BuildModel()
    scn.TrainModel(conf)
    #scn.TestModel(conf)
    #sess = scn.LoadModel()
    #scn.Evaluate(sess)
    #results = scn.BuildIndex(sess)
    #print(len(results))

    #scn.TrainModel()
