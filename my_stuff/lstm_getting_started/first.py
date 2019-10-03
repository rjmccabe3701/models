#!/usr/bin/python

'''
The purpose of this script to understand what the hell is going
on in the LSTM blocks.
'''
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ""
import tensorflow as tf
import numpy as np

USE_PLACEHOLDER_STATE=True

lstm = tf.compat.v1.nn.rnn_cell.LSTMCell(1)
x = tf.placeholder("float", [None, 1])
if USE_PLACEHOLDER_STATE:
    init_state1 = tf.placeholder("float", shape=(1,1))
    init_state2 = tf.placeholder("float", shape=(1,1))
    init_state = tf.nn.rnn_cell.LSTMStateTuple(c=init_state1, h=init_state2)
else:
    #  init_state = lstm.get_initial_state(batch_size=1, dtype=tf.float32)
    init_state = lstm.zero_state(batch_size=1, dtype=tf.float32)

result,next_state = lstm(x, init_state)
I = tf.global_variables_initializer()
sess = tf.Session()
sess.run(I)

feed_dict={x:np.array(3, ndmin=2)}
if USE_PLACEHOLDER_STATE:
    feed_dict[init_state1] = np.array(1, ndmin=2)
    feed_dict[init_state2] = np.array(2, ndmin=2)

OUTPUT = sess.run({'result': result, 'next_state': next_state}, feed_dict=feed_dict)
print("result = ", OUTPUT['result'], ", next_state = ", OUTPUT['next_state'])

#  tf.io.write_graph(sess.graph, '/var/tmp/tb_logs/lstm_first', 'train.pbtxt')
if USE_PLACEHOLDER_STATE:
    tf.compat.v1.summary.FileWriter('/var/tmp/tb_logs/lstm_first_placeholder', sess.graph)
else:
    tf.compat.v1.summary.FileWriter('/var/tmp/tb_logs/lstm_first', sess.graph)
"""
lstm.get_weights()
In [33]: sess.run(init_state)
Out[33]: LSTMStateTuple(c=array([[0.]], dtype=float32), h=array([[0.]], dtype=float32))

"""
