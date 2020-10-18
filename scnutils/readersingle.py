#coding=utf-8
import cPickle as pickle
import numpy as np
import random
def unison_shuffle(data, seed=None):
    if seed is not None:
        np.random.seed(seed)

    y = np.array(data['y'])
    c = np.array(data['c'])
    r = np.array(data['r'])

    assert len(y) == len(c) == len(r)
    p = np.random.permutation(len(y))
    shuffle_data = {'y': y[p], 'c': c[p], 'r': r[p]}
    return shuffle_data

def split_c(c, split_id):
    '''c is a list, example context  [int,int,int,eos,int,int,int,int,eos,int]->[[int,int,int],[int,int,int,int],[int]]
       split_id is a integer, conf[_EOS_] "_EOS_": 28270
       return nested list 嵌套列表
    '''
    turns = [[]]
    for _id in c: #_id就是具体的每一个wordid
        if _id != split_id:
            turns[-1].append(_id)
        else:
            turns.append([]) #turns = [[int,int,int],[]]
    if turns[-1] == [] and len(turns) > 1:
        turns.pop()
    return turns

def normalize_length(_list, length, cut_type='tail'):
    '''_list is a list or nested list, example turns/r/single turn c
       cut_type is head or tail, if _list len > length is used
       return a list len=length and min(read_length, length)
    '''
    real_length = len(_list) #每一轮对话长短不一
    if real_length == 0:
        return [0]*length, 0

    if real_length <= length: #length=max_turn_num=10
        if not isinstance(_list[0], list):              #不是嵌套类型，single list [int,int,int]
            _list.extend([0]*(length - real_length))    #[int,int,int,0,0,0,0,0,0,0]
        else:                                           #嵌套类型
			#_list=[[int,int,int],[int,int,int,int],[int]]
            _list.extend([[]]*(length - real_length)) #_list=[[int,int,int],[int,int,int,int],[int]，[],[],[],[]...] 7个[]
        return _list, real_length  #real_length=3

    if cut_type == 'head':#超过10的轮次，则留下开头100个
        return _list[:length], length
    if cut_type == 'tail': #留下后10个
        return _list[-length:], length

def produce_one_sample(data, index, split_id,  max_turn_len, turn_cut_type='tail', term_cut_type='tail'):
    '''max_turn_num=10
       max_turn_len=50
       return y, int 1 or 0
	   nor_turns_nor_c,[10,50]
	   nor_r,[50]
	   turn_len, int:10或者小于10的一个整数
	   term_len, [int,int,int.....]：元素为50或者小于50的一个整数，长度为10
	   r_len
    '''
    c = data['c'][index] #[int,int,int,eos,int,int,int,int,eos,int]
    r = data['r'][index][:]
    y = data['y'][index]
    c= [x for x in c if x != split_id]
    q_vec, q_len = normalize_length(c, max_turn_len, turn_cut_type)#对每一context进行统一，不足的补零，多了的掐头去尾
    nor_r, r_len = normalize_length(r, max_turn_len, term_cut_type)
    return q_vec, q_len, nor_r, r_len,y

def build_one_batch(data, batch_index, conf, turn_cut_type='tail', term_cut_type='tail'):
    target_loss_weights=[1,1]
    _q_vec = []

    _q_len = []

    _response = []
    _response_len = []

    _label = []
    _label_weight=[]
    for i in range(conf['batch_size']):#对每一个样本的处理
        index = batch_index * conf['batch_size'] + i
        q_vec, q_len, nor_r, r_len, y = produce_one_sample(data, index, conf['_EOS_'],
                conf['max_turn_len'], turn_cut_type, term_cut_type)

        _label.append(y)
        _q_vec.append(q_vec)
        _response.append(nor_r)
        _q_len.append(q_len)
      #  _tt_turns_len.append(turn_len)
        _response_len.append(r_len)
        if y > 0:  # pos weight=0
            _label_weight.append(target_loss_weights[1])
        else:
            _label_weight.append(target_loss_weights[0])

    return _q_vec,_q_len,_response, _response_len,_label,_label_weight
    

def build_batches(data, conf, turn_cut_type='tail', term_cut_type='tail',shuffle=True):
    _turns_batches = []
    _tt_turns_len_batches = []
    _every_turn_len_batches = []

    _response_batches = []
    _response_len_batches = []
    _label_weight_batches=[]
    _label_batches = []
    _label_weight=[]
    num_batches_per_epoch = len(data['y'])/conf['batch_size']
	# 50 0 000/256=1953 #舍掉后面的32个
  #  for epoch in range(conf["epoch"]):
       # if shuffle:
       #     shuffle_train = unison_shuffle(data)

       # else:#test=False
       #     shuffle_train =data
    for batch_index in range(num_batches_per_epoch):
        _q_vec, _q_len, _response, _response_len, _label,_label_weight = build_one_batch(data, batch_index, conf, turn_cut_type='tail', term_cut_type='tail')

        _turns_batches.append(_q_vec)
        _tt_turns_len_batches.append(_q_len)

        _response_batches.append(_response)
        _response_len_batches.append(_response_len)

        _label_batches.append(_label)
        _label_weight_batches.append(_label_weight)
    ans = {
                "turns": _turns_batches, "tt_turns_len": _tt_turns_len_batches,
                "response": _response_batches, "response_len": _response_len_batches, "label": _label_batches,
                "label_weight":_label_weight_batches
            }
    return ans



        


    








    
    


