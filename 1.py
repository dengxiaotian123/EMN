#coding=utf-8
#查看SCN本来的数据结构
#再确认一下
import os
os.environ['CUDA_VISIBLE_DEVICES']='0'

import tensorflow as tf
import pickle
train_data, val_data, test_data = pickle.load(open("../../data/douban/data.pkl", 'rb'))
#print(len(train_data['r']))
#print(len(val_data['r']))
print(test_data['r'][:50])
print(test_data['y'][:50])
print(test_data['c'][:50])
