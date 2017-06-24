# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
import tensorflow as tf


def inference(x_ph):
    hidden1 = tf.layers.dense(x_ph, 32)
    hidden1 = tf.nn.relu(hidden1)
    hidden2 = tf.layers.dense(hidden1, 32)
    hidden2 = tf.nn.relu(hidden2)
    logits = tf.layers.dense(hidden2, 3)
    y = tf.nn.softmax(logits)
    return y


np.random.seed(0)

df = pd.read_csv(os.path.join("data", "tictactoe.csv"))
x_train = df.iloc[:, :-1].values.astype(float)

y_train = np.zeros([len(df), 3])
for i, j in enumerate(df.iloc[:, -1].values):
    if j == 1:
        # x win
        y_train[i][0] = 1.
    elif j == -1:
        # o win
        y_train[i][1] = 1.
    else:
        # draw
        y_train[i][2] = 1.

with tf.Graph().as_default() as g:
    tf.set_random_seed(0)
    x_ph = tf.placeholder(tf.float32, [None, 9])
    y_ph = tf.placeholder(tf.float32, [None, 3])
    y = inference(x_ph)
    cross_entropy = -tf.reduce_mean(y_ph * tf.log(y))
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_ph, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    train_op = tf.train.AdamOptimizer(learning_rate=0.0002).minimize(cross_entropy)
    init_op = tf.global_variables_initializer()
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(init_op)
        for i in range(20000):
            ind = np.random.choice(len(y_train), 1000)
            sess.run(train_op, feed_dict={x_ph: x_train[ind], y_ph: y_train[ind]})
            if i % 100 == 0:
                train_loss = sess.run(cross_entropy, feed_dict={x_ph: x_train, y_ph: y_train})
                train_accuracy, y_pred = sess.run([accuracy, y], feed_dict={x_ph: x_train, y_ph: y_train})
                print("Iteration: {0} Loss: {1} Accuracy: {2}".format(i, train_loss, train_accuracy))
                # tf.logging.info(y_pred)
        if not os.path.isdir("checkpoints"):
            os.mkdir("checkpoints")
        saver.save(sess, "checkpoints/tictactoe")
        # Save model for deployment on ML Engine
        input_key = tf.placeholder(tf.int64, [None, ], name="key")
        output_key = tf.identity(input_key)
        input_signatures = {
            "key": tf.saved_model.utils.build_tensor_info(input_key),
            "x": tf.saved_model.utils.build_tensor_info(x_ph)
        }
        output_signatures = {
            "key": tf.saved_model.utils.build_tensor_info(output_key),
            "y": tf.saved_model.utils.build_tensor_info(y)
        }
        predict_signature_def = tf.saved_model.signature_def_utils.build_signature_def(
            input_signatures,
            output_signatures,
            tf.saved_model.signature_constants.PREDICT_METHOD_NAME
        )
        builder = tf.saved_model.builder.SavedModelBuilder(os.path.join("model"))
        builder.add_meta_graph_and_variables(
            sess,
            [tf.saved_model.tag_constants.SERVING],
            signature_def_map={
                tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: predict_signature_def
            },
            assets_collection=tf.get_collection(tf.GraphKeys.ASSET_FILEPATHS)
        )
        builder.save()
