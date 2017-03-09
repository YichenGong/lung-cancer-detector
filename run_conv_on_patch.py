from __future__ import print_function
import csv
import os
import numpy as np
import tensorflow as tf
import options
import importlib

from dataloader.candidates import CandidateDataLoader
from models.conv_on_patch_model import ConvOnPatches
log_dir = "oldLogs/"
model_dir = "conv-3/"
if not os.path.exists(log_dir + model_dir):
	os.makedirs(os.path.dirname(log_dir + model_dir))

opt = options.parse()
## args for loader
opt.top_k = 3
opt.diameter_mm = 30

#######################################################
# Build graph
#######################################################
with tf.device('/gpu:0'):
  chan = [1,32,64,64,128,128,256,256]
  kernel = [5,5,3,3,3,3,3]
  stride = [1,2,1,1,1,1,1]
  num_hidden = 64
  num_labels = 1
  num_nodules = opt.top_k

  tf_dataset = tf.placeholder(tf.float32, [num_nodules, None, opt.size[0], opt.size[1], opt.size[2], chan[0]])
  tf_labels = tf.placeholder(tf.float32, [None, num_labels])
  is_training = tf.placeholder(tf.bool)

  model = ConvOnPatches(num_nodules=num_nodules)
  logits = model.graph(tf_dataset, is_training, chan, kernel, stride, num_hidden, num_labels)
  loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits, tf_labels))

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
saver = tf.train.Saver()
global_min_loss = 100.0

with tf.Session(config=sess_config) as session:
  tf.global_variables_initializer().run()
  print("Initialized.")

  f = open(log_dir + model_dir + 'loss.log', 'w')
  for epoch in range(num_epochs):
    #################################################
    # Training
    #################################################
    dl.train()
    print('switch to training')
    for train_data, train_label, train_id in dl.data_iter():
      feed_dict = {tf_dataset: train_data, tf_labels: train_label, is_training: True}
      print('run the session now')
      _, l, preds = session.run([train_op, loss, prediction], feed_dict=feed_dict)
      #print('labels: preds \n %s' % np.concatenate((train_label, preds), axis=1))
      f.write('train: %f\n' % l)
      f.flush()

    #################################################
    # Validation
    #################################################
    dl.validate()
    total_loss = 0
    count = 0
    for valid_data, valid_label, valid_id in dl.data_iter():

      feed_dict = {tf_dataset: valid_data, tf_labels: valid_label, is_training: False}
      l = session.run(loss, feed_dict=feed_dict)
      batch_size = valid_id.shape[0]
      total_loss = total_loss + l * batch_size
      count = count + batch_size

    valid_loss = total_loss / count
    f.write('valid: %f\n' %  valid_loss)
    f.flush()

    if valid_loss < global_min_loss:
      # Saves the model and update global min loss
      print('update global min loss to: %f' % valid_loss)
      saver.save(session, log_dir + model_dir + 'model.ckpt')
      global_min_loss = valid_loss
  f.close()

  #################################################
  # Test Prediction
  #################################################
  dl.test()
  pred_dict = {}
  # Restore best model
  ckpt = tf.train.get_checkpoint_state(log_dir + model_dir)
  # print('checkpoint found: %s' % ckpt)
  # print('checkpoint path:%s' % ckpt.model_checkpoint_path)
  if ckpt and ckpt.model_checkpoint_path:
    saver.restore(session, ckpt.model_checkpoint_path)
    print('model restored.')

    for test_data, _, test_id in dl.data_iter():

      feed_dict = {tf_dataset: test_data, is_training: False}
      preds = session.run(prediction, feed_dict=feed_dict)
      for i in range(test_id.shape[0]):
        pred_dict[test_id[i][0]] = preds[i][0]

  print("Save submission to submission_backup.csv")
  with open(log_dir + model_dir + 'submission_backup.csv', 'w') as f:
    writer = csv.writer(f)
    # write the header
    for row in {'id': 'cancer'}.items():
      writer.writerow(row)
    # write the content
    for row in pred_dict.items():
      writer.writerow(row)
