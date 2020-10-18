from __future__ import print_function

import tensorflow as tf
def length(x):
    """
    :param x: tensor [256 50]
    :return:prem_seq_lengths (batch_size)
    mask_prem  (batch_size, max_seq_length, 1)
    """
   # mask_prem=tf.cast(tf.expand_dims(x, -1), tf.bool)  type=bool
    mask_prem = tf.cast(tf.cast(tf.expand_dims(x, -1), tf.bool), tf.float32) #type=float32
    return mask_prem
def masked_attention_axis1(x,mask):
    alph=tf.multiply(x,mask)
    alph_sum=tf.reduce_sum(alph,axis=2)
    output=tf.divide(alph,tf.expand_dims(alph_sum,axis=-1))



def masked_attention_axis2():
