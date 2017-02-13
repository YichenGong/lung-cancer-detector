from __future__ import print_function
import csv
import numpy as np
import tensorflow as tf

import sys
sys.path.append('./utils')

from load_data import DataLoad

flags = tf.app.flags
flags.DEFINE_integer("width", 128, "width")
flags.DEFINE_integer("height", 128, "height")
flags.DEFINE_integer("layers", 128, "layers")
flags.DEFINE_integer("batch_size", 10, "batch size")
flags.DEFINE_integer("num_process", 1, "process number")
flags.DEFINE_bool("is_train", True, "is train")
flags.DEFINE_string("data_type", "stage1", "sample or stage1")
config = flags.FLAGS

def Weight(shape, name):
    return tf.Variable(name=name + "_Weights", 
                       initial_value=tf.truncated_normal(shape=shape, mean=0, stddev=0.1))

def Bias(shape, name):
    return tf.Variable(name=name + "_Bias",
                      initial_value=tf.constant(shape=shape, value=0.0))

def conv(x, convShape, name):
    w = Weight(convShape, name)
    b = Bias([convShape[3]], name)
    return (tf.nn.conv2d(input=x, filter=w, strides=[1, 1, 1, 1],
                       padding='VALID',
                       name=name + "_2DConv") + b)

def pool(x, name):
    return tf.nn.max_pool(value=x, ksize=[1, 2, 2, 1], strides=[1, 1, 1, 1],
                         padding='SAME',
                         name=name + "_MaxPool")

# Parameters
batchSize = 10

imageSize = (128, 128, 128)
labelsSize = 1

#Making the Model here

#Make place for input
labelsInput = tf.placeholder(shape=[batchSize, labelsSize],
                            dtype=tf.float32,
                            name="InputLabels")

imagesPlaceholder = tf.placeholder(shape=[batchSize, imageSize[0], imageSize[1], imageSize[2]],
                                  dtype=tf.float32,
                                  name="InputImages")

#"Consume" the depth dimension
x_images = tf.reshape(imagesPlaceholder, [-1, imageSize[1], imageSize[2], 1])

#convolution Layes
hidden_Conv1 = tf.nn.relu(conv(x_images, [3, 3, 1, 16], "hidden_Conv1"))
hidden_Pool1 = pool(hidden_Conv1, "hidden_Conv1")

hidden_Conv2 = tf.nn.relu(conv(hidden_Pool1, [3, 3, 16, 16], "hidden_Conv2"))
hidden_Pool2 = pool(hidden_Conv2, "hidden_Conv2")

hidden_Conv3 = tf.nn.relu(conv(hidden_Pool2, [3, 3, 16, 32], "hidden_Conv3"))
hidden_Pool3 = pool(hidden_Conv3, "hidden_Conv2")

hidden_Conv4 = tf.nn.relu(conv(hidden_Pool3, [3, 3, 32, 32], "hidden_Conv4"))
hidden_Pool4 = pool(hidden_Conv4, "hidden_Conv4")

hidden_Conv5 = tf.nn.relu(conv(hidden_Pool4, [3, 3, 32, 64], "hidden_Conv5"))
hidden_Pool5 = pool(hidden_Conv5, "hidden_Conv5")

hidden_Conv6 = tf.nn.relu(conv(hidden_Pool5, [3, 3, 64, 64], "hidden_Conv6"))
hidden_Pool6 = pool(hidden_Conv6, "hidden_Conv6")

flattened_vector = tf.reshape(hidden_Pool6, shape=[hidden_Pool6.get_shape()[0].value, 
                                                   hidden_Pool6.get_shape()[1].value * 
                                                   hidden_Pool6.get_shape()[2].value *
                                                   hidden_Pool6.get_shape()[3].value])
vector_expanded = tf.expand_dims(flattened_vector, 1)
bring_back = tf.reshape(vector_expanded, shape=[batchSize, -1, vector_expanded.get_shape()[2].value])
added_around_instance = tf.reduce_sum(bring_back, 1)

hidden_Dense1_weights = Weight([added_around_instance.get_shape()[1].value, 64], "hidden_Dense1")
hidden_Dense1_bias = Bias([1, 64], "hidden_Dense1")

output_Dense2_weights = Weight([64, labelsSize], "output")
output_Dense2_bias = Bias([1, labelsSize], "output")

hidden = tf.nn.relu(tf.matmul(added_around_instance, hidden_Dense1_weights) + hidden_Dense1_bias)
output = tf.matmul(hidden, output_Dense2_weights) + output_Dense2_bias

loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(output, labelsInput))

# Optimizer.
optimizer = tf.train.AdamOptimizer(0.03)
grads = optimizer.compute_gradients(loss)
capped_gvs = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in grads]
train_op = optimizer.apply_gradients(capped_gvs)

# Prediction
prediction = tf.sigmoid(output)


# Training
num_epochs = 200
sess_config = tf.ConfigProto()
sess_config.gpu_options.allow_growth = True
sess_config.log_device_placement=True

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
      pred_dict[test_id[i]] = preds[i][0]

  # TODO: write the predictions to file.
  print("Now update csv")
  with open('submission_backup.csv', 'w') as f:
    writer = csv.writer(f)
    # write the header
    for row in {'id':'cancer'}.items():
      writer.writerow(row)
    # write the content
    for row in pred_dict.items():
      writer.writerow(row)
