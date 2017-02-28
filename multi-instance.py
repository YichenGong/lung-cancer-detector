import csv
import numpy as np
import tensorflow as tf
import options
import importlib

opt = options.parse()

opt.data = 'stage1' #Defaulting to stage1 data
dl = (importlib.import_module("dataloader." + opt.data)).get_data_loader(opt)
dl.load(opt)

def expand_last_dim(*input_data):
  res = []
  for in_data in input_data:
    res.append(np.expand_dims(in_data, axis=len(in_data.shape)))
  if len(res) == 1:
    return res[0]
  else:
    return res

def Weight(shape, name):
    return tf.Variable(name=name + "_Weights", 
                       initial_value=tf.truncated_normal(shape=shape, mean=0, stddev=0.1))

def Bias(shape, name):
    return tf.Variable(name=name + "_Bias",
                      initial_value=tf.constant(shape=shape, value=0.0))

def conv(x, convShape, name, strides=[1, 2, 2, 1]):
    w = Weight(convShape, name)
    b = Bias([convShape[3]], name)
    return (tf.nn.conv2d(input=x, filter=w, strides=strides,
                       padding='VALID',
                       name=name + "_2DConv") + b)

def pool(x, name):
    return tf.nn.max_pool(value=x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                         padding='VALID',
                         name=name + "_MaxPool")

# Parameters
batchSize = opt.batch

imageSize = opt.size
labelsSize = 1

#Making the Model here

#Make place for input
is_training = tf.placeholder(tf.bool)

labelsInput = tf.placeholder(shape=[None, labelsSize],
                            dtype=tf.float32,
                            name="InputLabels")

imagesPlaceholder = tf.placeholder(shape=[None, imageSize[0], imageSize[1], imageSize[2], 1],
                                  dtype=tf.float32,
                                  name="InputImages")

#"Consume" the depth dimension
x_images = tf.reshape(imagesPlaceholder, [-1, imageSize[1], imageSize[2], 1])

#convolution Layes
hidden_Conv1 = tf.nn.relu(tf.contrib.layers.batch_norm(conv(x_images, [3, 3, 1, 4], "hidden_Conv1"), is_training=is_training))
hidden_Pool1 = pool(hidden_Conv1, "hidden_Conv1")

hidden_Conv2 = tf.nn.relu(tf.contrib.layers.batch_norm(conv(hidden_Pool1, [3, 3, 4, 4], "hidden_Conv2"), is_training=is_training))
hidden_Pool2 = pool(hidden_Conv2, "hidden_Conv2")

hidden_Conv3 = tf.nn.relu(tf.contrib.layers.batch_norm(conv(hidden_Pool2, [3, 3, 4, 8], "hidden_Conv3"), is_training=is_training))
hidden_Pool3 = pool(hidden_Conv3, "hidden_Conv2")

flattened_vector = tf.reshape(hidden_Pool3, shape=[-1, 
                                                   hidden_Pool3.get_shape()[1].value * 
                                                   hidden_Pool3.get_shape()[2].value *
                                                   hidden_Pool3.get_shape()[3].value])
vector_expanded = tf.expand_dims(flattened_vector, 1)
bring_back = tf.reshape(vector_expanded, shape=[-1, imageSize[0], vector_expanded.get_shape()[2].value])
added_around_instance = tf.reduce_sum(bring_back, 1)

hidden_Dense1_weights = Weight([added_around_instance.get_shape()[1].value, 64], "hidden_Dense1")
hidden_Dense1_bias = Bias([1, 64], "hidden_Dense1")

output_Dense2_weights = Weight([64, labelsSize], "output")
output_Dense2_bias = Bias([1, labelsSize], "output")

hidden = tf.nn.relu(tf.matmul(added_around_instance, hidden_Dense1_weights) + hidden_Dense1_bias)
output = tf.matmul(hidden, output_Dense2_weights) + output_Dense2_bias

loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=output, labels=labelsInput))

# Optimizer.
optimizer = tf.train.AdamOptimizer(learning_rate=0.03, beta1=0.5)
grads = optimizer.compute_gradients(loss)
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
  train_op = optimizer.apply_gradients(grads)

# Prediction
prediction = tf.sigmoid(output)


# Training
num_epochs = 200
sess_config = tf.ConfigProto()
sess_config.gpu_options.allow_growth = True
sess_config.log_device_placement=False
sess_config.allow_soft_placement=True

with tf.Session(config=sess_config) as session:
  tf.global_variables_initializer().run()
  print('Initialized')

  for epoch in range(num_epochs):
    # Training
    dl.train()
    for x, y, _ in dl.data_iter():
      train_data, train_label = expand_last_dim(x, y)

      feed_dict = {imagesPlaceholder: train_data, labelsInput: train_label, is_training: True}
      _, l, preds = session.run([train_op, loss, prediction], feed_dict=feed_dict)
      #print('labels: preds \n %s' % np.concatenate((train_label, preds), axis=1))
      print('Mini-batch loss: %f' % l)


    # Validation
    dl.validate()
    total_loss = 0
    count = 0
    for x, y, _ in dl.data_iter():
      valid_data, valid_label = expand_last_dim(x, y)

      feed_dict = {imagesPlaceholder: valid_data, labelsInput: valid_label, is_training: False}
      l = session.run(loss, feed_dict=feed_dict)
      batch_size = valid_data.shape[0]
      total_loss = total_loss + l * batch_size
      count = count + batch_size

    valid_loss = total_loss / count
    print('Validation loss is: %f', valid_loss)


  # Test predictions
  dl.test()
  pred_dict = {}
  for x, _, test_id in dl.data_iter():
    test_data = expand_last_dim(x)
    
    feed_dict = {imagesPlaceholder : test_data, is_training: False}
    preds = session.run(prediction, feed_dict=feed_dict)
    for i in range(test_data.shape[0]):
      pred_dict[test_id[i]] = preds[i][0]

  # TODO: write the predictions to file.
  print("Now update csv")
  with open('submission_multi_instance_3.csv', 'w') as f:
    writer = csv.writer(f)
    # write the header
    for row in {'id':'cancer'}.items():
      writer.writerow(row)
    # write the content
    for row in pred_dict.items():
      writer.writerow(row)
