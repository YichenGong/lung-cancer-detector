import numpy as np
import dicom
import os
# import cv2
import tensorflow as tf


def get_3d_data_from_dicom(path):
  """
    Gets 3d data from DICOM file as (depth, width, height)
  """
  slices = [dicom.read_file(path + '/' + s) for s in os.listdir(path)]
  slices.sort(key=lambda x: int(x.InstanceNumber))
  print("loaded " + path)
  return np.stack([s.pixel_array for s in slices])


def preprocess(data, depth, height, width):
  """
    Preprocess a single 3d data
    1. Resize to a uniform size
    2. ...

  """
  if data.shape[0] > depth:
    data = data[:depth]
  else:
    data = data
    # TODO: fill with zeros.

  # This expansion add channel dimension as last dimension
  data = np.expand_dims(data, axis=len(data.shape))
  return data


def conv3d(data, w):
  return tf.nn.conv3d(data, w, strides=[1, 1, 1, 1, 1], padding='SAME')


def max_pool3d(data, depth_stride):
  return tf.nn.max_pool3d(data, [1, depth_stride, 2, 2, 1], [1, depth_stride, 2, 2, 1], padding='SAME')


# Parameters
batch_size = 1
num_labels = 1

in_depth = 120
in_height = 512
in_width = 512
in_channels = 1

filter_depth = 3
filter_height = 3
filter_width = 3

layer1_channels = 8
layer2_channels = 8

num_hidden = 64



def conv_layer:




# Graph

# graph = tf.Graph()
# with graph.as_default():
with tf.device('/cpu:0'):
  # Input data.

  tf_train_dataset = tf.placeholder(
    tf.float32, shape=(batch_size, in_depth, in_height, in_width, in_channels))
  tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))



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
    conv1 = conv3d(data, layer1_weights)
    conv1 = tf.Print(conv1, [tf.argmax(conv1, 1)], 'argmax after conv1 = ')  # print something with tf.Print
    conv1 = tf.nn.relu(conv1 + layer1_biases)
    pool1 = max_pool3d(conv1, 2)

    conv2 = conv3d(pool1, layer2_weights)
    conv2 = tf.nn.relu(conv2 + layer2_biases)
    pool2 = max_pool3d(conv2, 2)

    shape = pool2.get_shape().as_list()
    reshape = tf.reshape(pool2, [shape[0], shape[1] * shape[2] * shape[3] * shape[4]])

    hidden = tf.nn.relu(tf.matmul(reshape, layer3_weights) + layer3_biases)
    return tf.matmul(hidden, layer4_weights) + layer4_biases


  # Training computation.
  logits = model(tf_train_dataset)
  loss = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels))

  # Optimizer.
  optimizer = tf.train.GradientDescentOptimizer(0.05).minimize(loss)

  # Prediction
  train_prediction = tf.nn.softmax(logits)

num_steps = 10

# with tf.Session(graph=graph) as session:
with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as session:
  tf.global_variables_initializer().run()
  print('Initialized')

  # test on one data point
  data = get_3d_data_from_dicom('../data/sample/images/0a0c32c9e08cc2ea76a71649de56be6d')
  train_data = preprocess(data, in_depth, in_height, in_width)
  print("preprocessed")
  train_label = np.array([1])

  for step in range(num_steps):
    feed_dict = {tf_train_dataset: np.array([train_data]), tf_train_labels: np.array([train_label])}
    _, l, predictions = session.run([optimizer, loss, train_prediction], feed_dict=feed_dict)
    print('Minibatch loss at step %d: %f' % (step, l))
