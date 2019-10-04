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
NUM_UNITS=2

lstm = tf.compat.v1.nn.rnn_cell.LSTMCell(NUM_UNITS,
        use_peepholes=False)
x = tf.placeholder("float", [None, 1])
if USE_PLACEHOLDER_STATE:
    placeholder_states = [tf.placeholder("float", shape=(1,NUM_UNITS)) for i in range(2)]
    init_state = tf.nn.rnn_cell.LSTMStateTuple(
            c=placeholder_states[0], h=placeholder_states[1])
else:
    #  init_state = lstm.get_initial_state(batch_size=1, dtype=tf.float32)
    init_state = lstm.zero_state(batch_size=1, dtype=tf.float32)

result,next_state = lstm(x, init_state)

dummy_result = tf.add(result, 1.0)
dummy_next_state = tf.add(next_state, 2.0)


I = tf.global_variables_initializer()
sess = tf.Session()
sess.run(I)

feed_dict={x:np.array(3, ndmin=2)}
if USE_PLACEHOLDER_STATE:
    feed_dict[placeholder_states[0]] = np.array([[1 for _ in range(NUM_UNITS)]])
    feed_dict[placeholder_states[1]] = np.array([[2 for _ in range(NUM_UNITS)]])


OUTPUT = sess.run({'result': result, 'next_state': next_state}, feed_dict=feed_dict)
print("result = ", OUTPUT['result'], ", next_state = ", OUTPUT['next_state'])

#  tf.io.write_graph(sess.graph, '/var/tmp/tb_logs/lstm_first', 'train.pbtxt')
if USE_PLACEHOLDER_STATE:
    tf.compat.v1.summary.FileWriter('/var/tmp/tb_logs/lstm_first_placeholder_u{}'.format(NUM_UNITS), sess.graph)
else:
    tf.compat.v1.summary.FileWriter('/var/tmp/tb_logs/lstm_first', sess.graph)
"""
lstm.get_weights()
In [33]: sess.run(init_state)
Out[33]: LSTMStateTuple(c=array([[0.]], dtype=float32), h=array([[0.]], dtype=float32))


Notes:
https://medium.com/@erikhallstrm/hello-world-rnn-83cd7105b767
"""
