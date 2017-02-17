from __future__ import print_function
import csv
import os
import numpy as np
import tensorflow as tf
from utils.load_data import DataLoad

flags = tf.app.flags
flags.DEFINE_integer("width", 128, "width")
flags.DEFINE_integer("height", 128, "height")
flags.DEFINE_integer("layers", 128, "layers")
flags.DEFINE_integer("batch_size", 64, "batch size")
flags.DEFINE_integer("num_process", 1, "process number")
flags.DEFINE_bool("is_train", True, "is train")
flags.DEFINE_string("data_type", "stage1", "sample or stage1")
config = flags.FLAGS

# Dir to save the log
log_dir = "log/"
model_dir = "cnn7-1/"
os.makedirs(os.path.dirname(log_dir + model_dir), exist_ok=True)

def expand_last_dim(*input_data):
  res = []
  for in_data in input_data:
    res.append(np.expand_dims(in_data, axis=len(in_data.shape)))
  if len(res) == 1:
    return res[0]
  else:
    return res

def conv_bn_relu(input, kernel_shape, stride, bias_shape, is_training):
  weights = tf.get_variable("weights", kernel_shape, initializer=tf.random_normal_initializer(stddev=0.3))
  biases = tf.get_variable("biases", bias_shape, initializer=tf.constant_initializer(0.0))

  conv = tf.nn.conv3d(input, weights, strides=stride, padding='VALID')
  batch_norm = tf.contrib.layers.batch_norm(conv + biases, is_training=is_training)
  relu = tf.nn.relu(batch_norm)
  return relu

def fc_bn_relu(input, weight_shape, bias_shape, is_training):
  print(weight_shape)
  weights = tf.get_variable("weights", weight_shape, initializer=tf.random_normal_initializer(stddev=0.3))
  biases = tf.get_variable("biases", bias_shape, initializer=tf.constant_initializer(0.0))

  hidden = tf.matmul(input, weights)
  batch_norm = tf.contrib.layers.batch_norm(hidden + biases, is_training=is_training)
  relu = tf.nn.relu(batch_norm)
  return relu

def output_layer(input, weight_shape, bias_shape):
  weights = tf.get_variable("weights", weight_shape, initializer=tf.random_normal_initializer())
  biases = tf.get_variable("biases", bias_shape, initializer=tf.constant_initializer(0.0))
  return tf.matmul(input, weights) + biases

def dropout(input, keep_prob=0.8):
  return tf.nn.dropout(input, keep_prob=keep_prob, name='Dropout')

def flatten(input):
  shape = input.get_shape().as_list()
  batch_size = tf.shape(input)[0]
  flattened_size = shape[1] * shape[2] * shape[3] * shape[4]
  return tf.reshape(input, [batch_size, flattened_size])

# Parameters
chan0 = 1 # channels, 0 is input channel
chan1 = 16
chan2 = 16
chan3 = 32
chan4 = 64
chan5 = 128
chan6 = 128
chan7 = 256

keep_prob1 = 0.8
keep_prob2 = 0.6

num_hidden = 128
num_labels = 1

# Graph
with tf.device('/gpu:0'):
  # Input data.
  is_training = tf.placeholder(tf.bool)
  tf_dataset = tf.placeholder(
    tf.float32, [None, config.layers, config.height, config.width, chan0])
  tf_labels = tf.placeholder(tf.float32, [None, num_labels])

  # Model.
  def model(data, phase):
    with tf.variable_scope("conv1"):
      relu1 = conv_bn_relu(data, kernel_shape=[5,5,5,chan0,chan1], stride=[1,2,2,2,1], bias_shape=[chan1], is_training=phase)

    with tf.variable_scope("conv2"):
      relu2 = conv_bn_relu(relu1, kernel_shape=[3,3,3,chan1,chan2], stride=[1,1,1,1,1], bias_shape=[chan2], is_training=phase)

    with tf.variable_scope("conv3"):
      relu3 = conv_bn_relu(relu2, kernel_shape=[3,3,3,chan2,chan3], stride=[1,2,2,2,1], bias_shape=[chan3], is_training=phase)
 
    with tf.variable_scope("conv4"):
      relu4 = conv_bn_relu(relu3, kernel_shape=[3,3,3,chan3,chan4], stride=[1,1,1,1,1], bias_shape=[chan4], is_training=phase)
      relu4 = dropout(relu4, keep_prob1)

    with tf.variable_scope("conv5"):
      relu5 = conv_bn_relu(relu4, kernel_shape=[3,3,3,chan4,chan5], stride=[1,2,2,2,1], bias_shape=[chan5], is_training=phase)
      relu5 = dropout(relu5, keep_prob1)      

    with tf.variable_scope("conv6"):
      relu6 = conv_bn_relu(relu5, kernel_shape=[3,3,3,chan5,chan6], stride=[1,1,1,1,1], bias_shape=[chan6], is_training=phase)
      relu6 = dropout(relu6, keep_prob1)

    with tf.variable_scope("conv7"):
      relu7 = conv_bn_relu(relu6, kernel_shape=[3,3,3,chan6,chan7], stride=[1,2,2,2,1], bias_shape=[chan7], is_training=phase)
      relu7 = dropout(relu7, keep_prob2)

    with tf.variable_scope("fc"):
      reshape = flatten(relu7)
      hidden = fc_bn_relu(reshape, weight_shape=[reshape.get_shape().as_list()[1], num_hidden], bias_shape=[num_hidden], is_training=phase)
      hidden = dropout(hidden, keep_prob2)

    with tf.variable_scope("output"):
      return output_layer(hidden, weight_shape=[num_hidden, num_labels], bias_shape=[num_labels])

  logits = model(tf_dataset, is_training)
  # Prediction
  loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits, tf_labels))

  # Optimizer.
  update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
  with tf.control_dependencies(update_ops):
    # Ensures that we execute the update_ops before performing the train_step, for batch_norm
    train_op = tf.train.GradientDescentOptimizer(learning_rate=0.03).minimize(loss)

  # Prediction
  prediction = tf.sigmoid(logits)


# Training
num_epochs = 600
sess_config = tf.ConfigProto()
sess_config.gpu_options.allow_growth = True
sess_config.log_device_placement=False
sess_config.allow_soft_placement=True

saver = tf.train.Saver()
global_min_loss = 5.0 # initialize

with tf.Session(config=sess_config) as session:
  tf.global_variables_initializer().run()
  print("Initialized.")
  
  data_loader = DataLoad(config=config)
  f = open(log_dir + model_dir + 'loss.log', 'w')
  for epoch in range(num_epochs):
    print('epoch %d :' % epoch)
    # Training
    data_loader.train(uniform_distribution=True)
    while data_loader.has_next_batch():
      train_data, train_label, _ = data_loader.next_batch()
      train_data, train_label = expand_last_dim(train_data, train_label)
      
      feed_dict = {tf_dataset: train_data, tf_labels: train_label, is_training: True}
      _, l, preds = session.run([train_op, loss, prediction], feed_dict=feed_dict)
      # print('labels: preds \n %s' % np.concatenate((train_label, preds), axis=1)) 
      f.write('train: %f\n' % l)
      f.flush()

    # Validation
    data_loader.validation()
    total_loss = 0
    count = 0
    while data_loader.has_next_batch():
      valid_data, valid_label, _ = data_loader.next_batch()
      valid_data, valid_label = expand_last_dim(valid_data, valid_label)

      feed_dict = {tf_dataset: valid_data, tf_labels: valid_label, is_training: False}
      l = session.run(loss, feed_dict=feed_dict)
      batch_size = valid_data.shape[0]
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
  # Test predictions
  data_loader.test()
  pred_dict = {}
  # Restore best model
  ckpt = tf.train.get_checkpoint_state(log_dir + model_dir)
  print('checkpoint found: %s' % ckpt)
  print('checkpoint path:%s' % ckpt.model_checkpoint_path)
  if ckpt and ckpt.model_checkpoint_path:
    saver.restore(session, ckpt.model_checkpoint_path)
    print('model restored.')
  
    while data_loader.has_next_batch():
      test_data, _, test_id = data_loader.next_batch()
      test_data = expand_last_dim(test_data)
    
      feed_dict = {tf_dataset : test_data, is_training: False}
      preds = session.run(prediction, feed_dict=feed_dict)
      for i in range(test_data.shape[0]):
        pred_dict[test_id[i]] = preds[i][0]

  print("Save submission to submission_backup.csv")
  with open(log_dir + model_dir + 'submission_backup.csv', 'w') as f:
    writer = csv.writer(f)
    # write the header
    for row in {'id':'cancer'}.items():
      writer.writerow(row)
    # write the content
    for row in pred_dict.items():
      writer.writerow(row)
