from __future__ import print_function
import csv
import os
import numpy as np
import tensorflow as tf
import options
import importlib

from dataloader.candidates import CandidateDataLoader
from models.conv_on_patch_model import ConvOnPatches

opt = options.parse()
## args for loader
opt.top_k = 3
opt.diameter_mm = 30

#######################################################
# Build graph
#######################################################
with tf.device('/gpu:0'):
  chan = [1,4,8,16]
  num_hidden = 64
  num_labels = 1
  num_nodules = opt.top_k

  tf_dataset = tf.placeholder(tf.float32, [num_nodules, None, opt.size[0], opt.size[1], opt.size[2], chan[0]])
  tf_labels = tf.placeholder(tf.float32, [None, num_labels])
  is_training = tf.placeholder(tf.bool)

  model = ConvOnPatches(num_nodules=num_nodules)
  logits = model.graph(tf_dataset, is_training, chan, num_hidden, num_labels)
  loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=tf_labels))

  # Optimizer.
  update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
  with tf.control_dependencies(update_ops):
    # Ensures that we execute the update_ops before performing the train_step, for batch_norm
    train_op = tf.train.AdamOptimizer(learning_rate=0.01, beta1=0.7).minimize(loss)

  # Prediction
  prediction = tf.sigmoid(logits)

#######################################################
# Training Session
#######################################################
num_epochs = opt.epochs
sess_config = tf.ConfigProto()
sess_config.gpu_options.allow_growth = True
sess_config.log_device_placement=False
sess_config.allow_soft_placement=True

dl = CandidateDataLoader(opt)

with tf.Session(config=sess_config) as session:
  tf.global_variables_initializer().run()
  print("Initialized.")

  for epoch in range(num_epochs):
    # Training
    dl.train()
    print('switch to training')
    for train_data, train_label in dl.data_iter():

      feed_dict = {tf_dataset: train_data, tf_labels: train_label, is_training: True}
      print('run the session now')
      _, l, preds = session.run([train_op, loss, prediction], feed_dict=feed_dict)
      #print('labels: preds \n %s' % np.concatenate((train_label, preds), axis=1))
      print('batch loss:{}'.format(l))   
   

    # Validation
    dl.validate()
    total_loss = 0
    count = 0
    for valid_data, valid_label in dl.data_iter():

      feed_dict = {tf_dataset: valid_data, tf_labels: valid_label, is_training: False}
      l = session.run(loss, feed_dict=feed_dict)
      batch_size = valid_data.shape[0]
      total_loss = total_loss + l * batch_size
      count = count + batch_size

    valid_loss = total_loss / count
    print('epoch[{}] valid loss: {}'.format(epoch, valid_loss))
