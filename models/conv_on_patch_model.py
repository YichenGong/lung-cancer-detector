import tensorflow as tf

class ConvOnPatches():
  def __init__(self, num_nodules):
    self.k = num_nodules

  def graph(self, data, phase, chan, kernel, stride, num_hidden, num_labels):
    # build the conv layer for k nodules sharing weights, conv[layer_idx][nodule_idx]
    conv = [data] # init the layer zero as data
    num_conv_layers = len(chan) - 1

    for layer in range(1, num_conv_layers + 1):
      conv.append([])
      for i in range(self.k):
        with tf.variable_scope("conv_{}".format(layer), reuse=(i>0)):
          conv[layer].append(conv_bn_relu(conv[layer - 1][i],
                                    kernel_shape=[kernel[layer - 1], kernel[layer - 1], kernel[layer - 1], chan[layer - 1], chan[layer]],
                                    stride=[1, stride[layer - 1], stride[layer - 1], stride[layer - 1], 1],
                                    bias_shape=[chan[layer]],
                                    is_training=phase))
          conv[layer][i] = dropout(conv[layer][i], keep_prob=0.6, is_training=phase)

    hidden = []
    for i in range(self.k):
      with tf.variable_scope("fc", reuse=(i>0)):
        reshape = flatten(conv[num_conv_layers][i])
        hidden.append(fc_bn_relu(reshape,
                                 weight_shape=[reshape.get_shape()[1].value, num_hidden],
                                 bias_shape=[num_hidden],
                                 is_training=phase))
        hidden[i] = dropout(hidden[i], keep_prob=0.5, is_training=phase)

    output = []
    for i in range(self.k):
      with tf.variable_scope("output", reuse=(i>0)):
        output.append(output_layer(hidden[i],
                                   weight_shape=[num_hidden, num_labels],
                                   bias_shape=[num_labels]))

    return tf.reduce_max(output, axis=0)



def conv_bn_relu(input, kernel_shape, stride, bias_shape, is_training):
  weights = tf.get_variable("weights", kernel_shape, initializer=tf.random_normal_initializer(stddev=0.3))
  biases = tf.get_variable("biases", bias_shape, initializer=tf.constant_initializer(0.0))

  conv = tf.nn.conv3d(input, weights, strides=stride, padding='SAME') + biases
  batch_norm = tf.contrib.layers.batch_norm(conv, is_training=is_training)
  relu = tf.nn.relu(batch_norm)
  return relu

def fc_bn_relu(input, weight_shape, bias_shape, is_training=False):
  weights = tf.get_variable("weights", weight_shape, initializer=tf.random_normal_initializer(stddev=0.3))
  biases = tf.get_variable("biases", bias_shape, initializer=tf.constant_initializer(0.0))

  hidden = tf.matmul(input, weights) + biases
  batch_norm = tf.contrib.layers.batch_norm(hidden, is_training=is_training)
  relu = tf.nn.relu(batch_norm)
  return relu

def output_layer(input, weight_shape, bias_shape):
  weights = tf.get_variable("weights", weight_shape, initializer=tf.random_normal_initializer())
  biases = tf.get_variable("biases", bias_shape, initializer=tf.constant_initializer(0.0))
  return tf.matmul(input, weights) + biases

def dropout(input, keep_prob=0.8, is_training=False):
  prob = tf.cond(is_training, lambda: tf.constant(keep_prob), lambda: tf.constant(1.0))
  return tf.nn.dropout(input, keep_prob=prob, name='Dropout')

def flatten(input):
  shape = input.get_shape().as_list()
  batch_size = tf.shape(input)[0]
  flattened_size = shape[1] * shape[2] * shape[3] * shape[4]
  return tf.reshape(input, [batch_size, flattened_size])
