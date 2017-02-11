from __future__ import print_function
import numpy as np
import tensorflow as tf
from load_data import DataLoad

flags = tf.app.flags
flags.DEFINE_integer("width", 128, "width")
flags.DEFINE_integer("height", 128, "height")
flags.DEFINE_integer("layers", 128, "layers")
flags.DEFINE_integer("batch_size", 20, "batch size")
flags.DEFINE_integer("num_process", 1, "process number")
flags.DEFINE_bool("is_train", True, "is train")
flags.DEFINE_string("data_type", "stage1", "sample or stage1")
config = flags.FLAGS

def expand_last_dim(*input_data):
  res = []
  for in_data in input_data:
    res.append(np.expand_dims(in_data, axis=len(in_data.shape)))
  if len(res) == 1:
    return res[0]
  else:
    return res

def conv3d(input_data, w, stride):
  return tf.nn.conv3d(input_data, w, strides=stride, padding='SAME')

def max_pool3d(input_data, depth_stride):
  return tf.nn.max_pool3d(input_data, [1, depth_stride, 2, 2, 1], [1, depth_stride, 2, 2, 1], padding='SAME')

# Parameters
in_depth = config.layers
in_height = config.height
in_width = config.width
in_channels = 1

filter_depth = 3
filter_height = 3
filter_width = 3
conv_stride = [1, 1, 1, 1, 1]

layer1_channels = 8
layer2_channels = 32

num_hidden = 64
num_labels = 1

# Graph
with tf.device('/gpu:0'):
  # Input data.
  tf_dataset = tf.placeholder(
    tf.float32, [None, in_depth, in_height, in_width, in_channels])
  tf_labels = tf.placeholder(tf.float32, [None, num_labels])

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

  logits = model(tf_dataset)
  # Prediction
  loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits, tf_labels))

  # Optimizer.
  optimizer = tf.train.AdamOptimizer(0.03)
  grads = optimizer.compute_gradients(loss)
  capped_gvs = [(tf.clip_by_value(grad, -10., 10.), var) for grad, var in grads]
  train_op = optimizer.apply_gradients(capped_gvs)

  # Prediction
  prediction = tf.sigmoid(logits)


# Training
num_epochs = 1
sess_config = tf.ConfigProto()
sess_config.gpu_options.allow_growth = True
sess_config.log_device_placement=False

with tf.Session(config=sess_config) as session:
  tf.global_variables_initializer().run()
  print('Initialized')
  
  data_loader = DataLoad(config=config)

  for epoch in range(num_epochs):
    # Training
    data_loader.train()
    while data_loader.has_next_batch():
      train_data, train_label, _ = data_loader.next_batch()
      train_data, train_label = expand_last_dim(train_data, train_label)

      feed_dict = {tf_dataset: train_data, tf_labels: train_label}
      _, l, preds = session.run([train_op, loss, prediction], feed_dict=feed_dict)
      print('labels: preds \n %s' % np.concatenate((train_label, preds), axis=1))
      print('Mini-batch loss: %f' % l)


    # Validation
    data_loader.validation()
    total_loss = 0
    count = 0
    while data_loader.has_next_batch():
      valid_data, valid_label, _ = data_loader.next_batch()
      valid_data, valid_label = expand_last_dim(valid_data, valid_label)

      feed_dict = {tf_dataset: valid_data, tf_labels: valid_label}
      l = session.run(loss, feed_dict=feed_dict)
      batch_size = valid_data.shape[0]
      total_loss = total_loss + l * batch_size
      count = count + batch_size

    valid_loss = total_loss / count
    print('Validation loss is: %f', valid_loss)


  # Test predictions
  data_loader.test()
  pred_dict = {}
  while data_loader.has_next_batch():
    test_data, _, test_id = data_loader.next_batch()
    test_data = expand_last_dim(test_data)
    
    feed_dict = {tf_dataset : test_data}
    preds = session.run(prediction, feed_dict=feed_dict)
    for i in range(test_data.shape[0]):
      pred_dict[test_id[i]] = preds[i]

  # TODO: write the predictions to file.
  print("Now update csv")




