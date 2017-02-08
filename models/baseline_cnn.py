from __future__ import print_function
import numpy as np
import dicom
import os
import sys
import cv2
import tensorflow as tf
from tensorflow.python.client import timeline
import pandas as pd




def get_data_with_ids(folder, patient_ids):
  data = [get_3d_data_from_dicom(folder + patient_id) for patient_id in patient_ids]
  return np.array([preprocess(single_slice, in_depth, in_height, in_width) for single_slice in data])

def get_label_with_ids(label_path, patient_ids):
  label_file = pd.read_csv(label_path, index_col=0).T.to_dict()
  labels = {key: value['cancer'] for key, value in label_file.items()}
  label_array =  np.array([labels[patient_id] for patient_id in patient_ids])
  return np.expand_dims(label_array, axis=len(label_array.shape))

def get_3d_data_from_dicom(path):
  """
    Gets 3d data from DICOM file as (depth, width, height)
  """
  slices = [dicom.read_file(path + '/' + s) for s in os.listdir(path)]
  slices.sort(key=lambda x: int(x.InstanceNumber))
  return np.stack([s.pixel_array for s in slices])

def preprocess(scan, depth, height, width):
  """
    Preprocess a single 3d data
    1. Resize to a uniform size
    2. ...

  """

  # change depth.
  if scan.shape[0] > depth:
    scan = scan[:depth]
  else:
    scan = scan
    # TODO: fill with air

  # resize width and height
  scan = np.array([cv2.resize(single_slice, (height, width)) for single_slice in scan])

  # This expansion add channel dimension as last dimension
  scan = np.expand_dims(scan, axis=len(scan.shape))
  return scan


def conv3d(input_data, w, stride):
  return tf.nn.conv3d(input_data, w, strides=stride, padding='SAME')


def max_pool3d(input_data, depth_stride):
  return tf.nn.max_pool3d(input_data, [1, depth_stride, 2, 2, 1], [1, depth_stride, 2, 2, 1], padding='SAME')


# Parameters
batch_size = 20
num_labels = 1

in_depth = 100
in_height = 128
in_width = 128
in_channels = 1

filter_depth = 3
filter_height = 3
filter_width = 3
conv_stride = [1, 1, 1, 1, 1]

layer1_channels = 8
layer2_channels = 32

num_hidden = 64

# Graph

# graph = tf.Graph()
# with graph.as_default():
with tf.device('/gpu:0'):
  # Input data.

  tf_train_dataset = tf.placeholder(
    tf.float32, [None, in_depth, in_height, in_width, in_channels])
  tf_train_labels = tf.placeholder(tf.float32, [None, num_labels])

  # Variables.
  layer1_weights = tf.Variable(tf.truncated_normal(
    [filter_depth, filter_height, filter_width, in_channels, layer1_channels], stddev=0.1))
  layer1_biases = tf.Variable(tf.zeros([layer1_channels]))

  layer2_weights = tf.Variable(tf.truncated_normal(
    [filter_depth, filter_height, filter_width, layer1_channels, layer2_channels], stddev=0.1))
  layer2_biases = tf.Variable(tf.constant(1.0, shape=[layer2_channels]))

  layer3_weights = tf.Variable(tf.truncated_normal(
    [in_depth // 4 * in_height // 4 * in_width // 4 * layer2_channels, num_hidden], stddev=0.1))
  layer3_biases = tf.Variable(tf.constant(1.0, shape=[num_hidden]))

  layer4_weights = tf.Variable(tf.truncated_normal(
    [num_hidden, num_labels], stddev=0.1))
  layer4_biases = tf.Variable(tf.constant(1.0, shape=[num_labels]))


  # Model.
  def model(data):
    # normalize
    max_v = tf.reduce_max(data)
    data = tf.realdiv(data, max_v)

    conv1 = conv3d(data, layer1_weights, conv_stride)
    conv1 = tf.nn.relu(conv1 + layer1_biases)
    pool1 = max_pool3d(conv1, 2)

    conv2 = conv3d(pool1, layer2_weights, conv_stride)
    conv2 = tf.nn.relu(conv2 + layer2_biases)
    pool2 = max_pool3d(conv2, 2)

    shape = pool2.get_shape().as_list()
    reshape = tf.reshape(pool2, [tf.shape(data)[0], shape[1] * shape[2] * shape[3] * shape[4]])

    hidden = tf.nn.relu(tf.matmul(reshape, layer3_weights) + layer3_biases)
    return tf.matmul(hidden, layer4_weights) + layer4_biases


  # Training computation.
  logits = model(tf_train_dataset)
  # Prediction
  loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits, tf_train_labels))

  # Optimizer.
  optimizer = tf.train.GradientDescentOptimizer(0.05)
  grads = optimizer.compute_gradients(loss)
  capped_gvs = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in grads]
  train_op = optimizer.apply_gradients(capped_gvs)

  # Prediction
  train_prediction = tf.sigmoid(logits)

num_steps = 1000

# with tf.Session(graph=graph) as session:
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.log_device_placement=False

with tf.Session(config=config) as session:
  run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
  run_metadata = tf.RunMetadata()

  tf.global_variables_initializer().run()
  print('Initialized')

  # get patient_ids
  data_folder_path = '../data/sample/images/'
  label_file_path = '../data/stage1_labels.csv'
  ids = os.listdir(data_folder_path)
  ids = [item for item in ids if not item.startswith(".")]
  label_file = pd.read_csv(label_file_path, index_col=0).T.to_dict()
  labels = {key: value['cancer'] for key, value in label_file.items()}
  ids = [p_id for p_id in ids if p_id in labels.keys()]
  num_sample = len(ids)
  print('Total Number of data points: %d' % (num_sample))

  offset = 0
  for step in range(num_steps):
    batch_ids = ids[offset: min(offset + batch_size, num_sample)]
    offset = min(offset + batch_size, num_sample) % num_sample

    train_data = get_data_with_ids(data_folder_path, batch_ids)
    train_label = get_label_with_ids(label_file_path, batch_ids)

    feed_dict = {tf_train_dataset: np.array(train_data), tf_train_labels: np.array(train_label)}
    _, l, predictions = session.run([train_op, loss, train_prediction],
                                    feed_dict=feed_dict,
                                    options=run_options,
                                    run_metadata=run_metadata)

    print('Minibatch loss at step %d: %f' % (step, l))

    # Create the Timeline object, and write it to a json
    # go to the page 'chrome://tracing' and load the timeline.json file
    tl = timeline.Timeline(run_metadata.step_stats)
    ctf = tl.generate_chrome_trace_format()
    with open('timeline.json', 'w') as f:
      f.write(ctf)
