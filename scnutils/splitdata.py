#coding=utf-8
import cPickle as pickle
import numpy as np
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
def produce_one_sample(data,  split_id=28270, max_turn_num=10, max_turn_len=50, turn_cut_type='tail', term_cut_type='tail'):
    '''max_turn_num=10
       max_turn_len=50
       return y, int 1 or 0
	   nor_turns_nor_c,[10,50]
	   nor_r,[50]
	   turn_len, int:10或者小于10的一个整数
	   term_len, [int,int,int.....]：元素为50或者小于50的一个整数，长度为10
	   r_len
    '''
    data_new={}
    for index in range(len(data['y'])):
        c = data['c'][index]#[int,int,int,eos,int,int,int,int,eos,int]
        r = data['r'][index][:]
        y = data['y'][index]
        turns = split_c(c, split_id)#[[int,int,int],[int,int,int,int],[int]]   #转换为嵌套列表
        #normalize turns_c length, nor_turns length is max_turn_num
        nor_turns, turn_len = normalize_length(turns, max_turn_num, turn_cut_type)#对每一context进行统一，不足的补零，多了的掐头去尾
        if turn_len<=10:

        nor_turns_nor_c = []
        term_len = []
        #nor_turn_nor_c length is max_turn_num, element is a list length is max_turn_len
        for c in nor_turns:
            #nor_c length is max_turn_len
            nor_c, nor_c_len = normalize_length(c, max_turn_len, term_cut_type) #对context中的每一句子（terms）进行统一，不足的补零，多了的掐头去尾
            nor_turns_nor_c.append(nor_c)
            term_len.append(nor_c_len) #记录 real_length

        nor_r, r_len = normalize_length(r, max_turn_len, term_cut_type)

    return y, nor_turns_nor_c, nor_r, turn_len, term_len, r_len
