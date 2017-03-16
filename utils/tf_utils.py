import tensorflow as tf

def conv_2d_drop_bn_relu(inp, inp_chan, out_chan, kernel, stride=1, prob=1.0, name=""):
	weights = tf.Variable(tf.truncated_normal(
			shape=[kernel, kernel, inp_chan, out_chan],
			mean=0.0,
			stddev=1.0),
		name=name+"_weights")
	bias = tf.Variable(tf.constant(
			shape=[out_chan],
			value=0.0),
		name=name+"_bias")
	conv = tf.nn.conv2d(input=inp,
		filter=weights,
		strides=[1, stride, stride, 1],
		padding='VALID',
		name=name+"_conv")
	drop = tf.nn.dropout(conv, prob, name=name+"_drop")
	out = tf.nn.relu(tf.contrib.layers.batch_norm(drop + bias))

	return out, weights, bias

def pool_2d(inp, kernel, stride, name=""):
	out = tf.nn.max_pool(
		value=inp, 
		ksize=[1, kernel, kernel, 1],
		strides=[1, stride, stride, 1],
		padding='VALID',
		name=name+"_pool")
	return out

def fc_drop_bn_relu(inp, inp_size, out_size, prob=1.0, name=""):
	weights = tf.Variable(tf.truncated_normal(
			shape=[inp_size, out_size],
			mean=0.0,
			stddev=1.0),
		name=name+"_weights")
	bias = tf.Variable(tf.constant(
			shape=[out_size],
			value=0.0),
		name=name+"_bias")

	out = tf.nn.relu(
			tf.contrib.layers.batch_norm(
				tf.nn.dropout(
					tf.matmul(inp, weights) + bias,
					prob, name=name+"_drop")))

	return out, weights, bias

def deconv_2d_drop_bn_relu(inp, inp_chan, out_chan, kernel, stride=1, prob=1.0, name=""):
	weights = tf.Variable(tf.truncated_normal(
			shape=[kernel, kernel, out_chan, inp_chan],
			mean=0.0,
			stddev=1.0),
		name=name+"_weights")
	bias = tf.Variable(tf.constant(
			shape=[out_chan],
			value=0.0),
		name=name+"_bias")

	inp_shape = tf.shape(inp)
	deconv = tf.nn.conv2d_transpose(
		value=inp,
		filter=weights,
		output_shape=tf.stack([inp_shape[0], inp_shape[1]*stride, inp_shape[2]*stride, out_chan]),
		strides=[1, stride, stride, 1],
		padding='VALID',
		name=name+"_deconv")

	drop = tf.nn.dropout(deconv, prob, name=name+"_drop")
	out = tf.nn.relu(tf.contrib.layers.batch_norm(drop + bias))

	return out, weights, bias

def add_weights_summary(weights, name=""):
	with tf.name_scope(name+"_summary"):
		mean = tf.reduce_mean(weights)
		tf.summary.scalar('mean', mean)
		with tf.name_scope('stddev'):
			stddev = tf.sqrt(tf.reduce_mean(tf.square(weights - mean)))
		tf.summary.scalar('stddev', stddev)
		tf.summary.scalar('max', tf.reduce_max(weights))
		tf.summary.scalar('min', tf.reduce_min(weights))
		tf.summary.histogram('histogram', weights)

def add_weights_as_images_summary(weights, height, width, channels, num=10, name=""):
	with tf.name_scope(name+"_summary"):
		weight_reshaped_as_image = tf.reshape(weights, [-1, height, width, channels])
		tf.summary.image('image',
			weight_reshaped_as_image,
			max_outputs=num)

def add_scalar_summary(val, name=""):
	with tf.name_scope(name+"_summary"):
		tf.summary.scalar('val', val)