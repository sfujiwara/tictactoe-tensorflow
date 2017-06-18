# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
import tictactoe

with tf.Graph().as_default() as g:
    sess = tf.Session()
    meta_graph = tf.saved_model.loader.load(
        sess=sess,
        tags=[tf.saved_model.tag_constants.SERVING],
        export_dir="model"
    )
    model_signature = meta_graph.signature_def[tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
    input_signature = model_signature.inputs
    output_signature = model_signature.outputs
    # Get names of input and output tensors
    input_tensor_name = input_signature["x"].name
    output_tensor_name = output_signature["y"].name
    # Get input and output tensors
    x_ph = sess.graph.get_tensor_by_name(input_tensor_name)
    y = sess.graph.get_tensor_by_name(output_tensor_name)
    print sess.run(y, feed_dict={x_ph: np.zeros([1, 9])})

env = tictactoe.TicTacToeEnv()
observation = env.reset()
for _ in range(9):
    env.render()
    # Compute scores
    scores = np.zeros(9)
    for i in range(9):
        if env.board[i] == 0:
            board_copy = np.array([env.board])
            board_copy[0][i] = 1
            prob = sess.run(y, feed_dict={x_ph: board_copy})
            print i, prob
            scores[i] = prob[0][0]/prob[0][1]
    print scores
    env.step(scores.argmax())
    env.render()
    player_move = input()
    env.step(player_move)
