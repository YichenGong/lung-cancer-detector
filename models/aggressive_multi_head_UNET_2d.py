from __future__ import print_function
import tensorflow as tf
import numpy as np
import utils.tf_utils as tfu

class MultiHeadUnet_2D:
	def __init__(self, config, image_size=(512, 512)):
		self.config = config

		self.create_inputs(image_size)
		self.build_encoder()

		self.create_nodule_segment_head()
		self.create_nodule_segment_loss()

		self.create_cancer_classification_head()
		self.create_cancer_classification_loss()

	def build_encoder(self):
		print("Creating encoder part...")
	
		#TODO make this into a simple loop
		#contains list of (weights, bias, output) generally
		#may change according to the type of layer	
		current_input = self.X

		current_channel = 1
		conv_kernels = [3, 3, 3, 3, 3, 3, 3, 3, 3, 3]
		conv_strides = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
		conv_channels = [64, 64, 128, 128, 256, 256, 512, 512, 1024, 1024]
		dropout_prob = self.drop_prob
		pool_kernel = [0, 2, 0, 2, 0, 2, 0, 2, 0, 0]

		self._encode_conv_weights = []
		self._encode_conv = []
		self._encode_pool = []

		for idx in range(len(conv_kernels)):
			level = (idx//2) + 1
			with tf.name_scope('encode_{}'.format(level)):
				out, weight, bias = tfu.conv_2d_drop_bn_relu(
						current_input,
						current_channel,
						conv_channels[idx],
						conv_kernels[idx],
						conv_strides[idx],
						dropout_prob,
						"{}".format(idx%2)
					)
				self._encode_conv_weights.append((weight, bias))
				self._encode_conv.append(out)

				if pool_kernel[idx]:
					out = tfu.pool_2d(
								out,
								pool_kernel[idx],
								pool_kernel[idx],
								"{}".format(idx%2)
							)
					self._encode_pool.append(out)
				current_input = out
				current_channel = conv_channels[idx]

		self._encode = out
		self._encode_features = current_channel
		print("Created encoder part!")

	def create_nodule_segment_head(self):
		'''
		We'll have more aggressive Deconvolution
		as compared to what is given in the U-NET

		Two 4x4 Deconvolutions with skip connections
		and last layer of 1x1 Convolution
		'''
		print("Creating Nodule Segmentation part...")

		self._nodule_seg_weights = []
		self._nodule_seg_outs = []
		
		current_input = self._encode
		current_channel = self._encode_features

		with tf.name_scope("Nodule_Segment_1"):
			out, weights, bias = tfu.deconv_2d_drop_bn_relu(
					current_input,
					current_channel,
					current_channel//4,
					4,
					4,
					self.drop_prob,
					"up_1"
				)
			self._nodule_seg_outs.append(out)
			self._nodule_seg_weights.append((weights, bias))

			x1_shape = tf.shape(self._encode_conv[5])
			x2_shape = tf.shape(out)
			# offsets for the top left corner of the crop
			offsets = [0, (x1_shape[1] - x2_shape[1]) // 2, (x1_shape[2] - x2_shape[2]) // 2, 0]
			size = [-1, x2_shape[1], x2_shape[2], -1]
			x1_crop = tf.slice(self._encode_conv[5], offsets, size)
			out = self._nodule_seg_skip_1 = tf.add(x1_crop, out)

			current_input = out
			current_channel = current_channel//4

			out, weights, bias = tfu.conv_2d_drop_bn_relu(
					current_input,
					current_channel,
					current_channel,
					3,
					1,
					self.drop_prob,
					"up_1"
				)
			self._nodule_seg_outs.append(out)
			self._nodule_seg_weights.append((weights, bias))
			current_input = out

		with tf.name_scope("Nodule_Segment_2"):
			out, weights, bias = tfu.deconv_2d_drop_bn_relu(
					current_input,
					current_channel,
					current_channel//4,
					4,
					4,
					self.drop_prob,
					"up_1"
				)
			self._nodule_seg_outs.append(out)
			self._nodule_seg_weights.append((weights, bias))

			x1_shape = tf.shape(self._encode_conv[1])
			x2_shape = tf.shape(out)
			# offsets for the top left corner of the crop
			offsets = [0, (x1_shape[1] - x2_shape[1]) // 2, (x1_shape[2] - x2_shape[2]) // 2, 0]
			size = [-1, x2_shape[1], x2_shape[2], -1]
			x1_crop = tf.slice(self._encode_conv[1], offsets, size)
			out = self._nodule_seg_skip_2 = tf.add(x1_crop, out)

			current_input = out
			current_channel = current_channel//4

			out, weights, bias = tfu.conv_2d_drop_bn_relu(
					current_input,
					current_channel,
					current_channel,
					3,
					1,
					self.drop_prob,
					"up_2"
				)
			self._nodule_seg_outs.append(out)
			self._nodule_seg_weights.append((weights, bias))
			current_input = out

		with tf.name_scope("Nodule_Segment_3"):
			out, weights, bias = tfu.conv_2d_drop_bn_relu(
					current_input,
					current_channel,
					1,
					1,
					1,
					self.drop_prob,
					"out"
				)
			self._nodule_seg_outs.append(out)
			self._nodule_seg_weights.append((weights, bias))

			self._nodule = out

		print("Created Nodule Segmentation part!")

	def create_nodule_segment_loss(self):
		print("Creating loss of Nodule Segmentation...")
		self._nodule_padded = tf.image.resize_images(self._nodule, 
			[self.Y_nodule.get_shape()[1].value, self.Y_nodule.get_shape()[2].value])
		logits_flattened = tf.reshape(self._nodule_padded, [-1, 1])
		target_flattened = tf.reshape(self.Y_nodule, [-1, 1])
		self._nodule_loss = tf.reduce_mean(
				tf.nn.weighted_cross_entropy_with_logits(
						targets=target_flattened,
						logits=logits_flattened,
						pos_weight=self.Y_nodule_weight,
						name="Nodule_Loss"
					)
			)

		self._nodule_gs = tf.Variable(0, trainable=False)
		self._nodule_lr = tf.train.exponential_decay(
			self.config.learning_rate, 
			self._nodule_gs, 
			50000, 
			self.config.decay_rate, 
			staircase=True)

		self._nodule_optimizer = tf.train.MomentumOptimizer(
			self._nodule_lr, 
			self.config.momentum).minimize(
				self._nodule_loss,
				global_step=self._nodule_gs)
		print("Creating loss of Nodule Segmentation...")

	def create_cancer_classification_head(self):
		'''
		Cancer classification head
		This will be a simple classification head 
		of cancer or not
		'''
		print("Creating Classification part...")

		out = self._encode
		channels = self._encode_features

		conv_channels = [256, 64]
		conv_kernels = [3, 3]
		conv_strides = [1, 1]

		self._cancer_weights = []
		self._cancer_outs = []

		with tf.name_scope("Cancer"):
			out, weights, bias = tfu.conv_2d_drop_bn_relu(
					out,
					channels,
					conv_channels[0],
					conv_kernels[0],
					conv_strides[0],
					self.drop_prob,
					name="classify_1"
				)
			self._cancer_weights.append((weights, bias))
			self._cancer_outs.append(out)
			channels = conv_channels[0]

			out, weights, bias = tfu.conv_2d_drop_bn_relu(
					out,
					channels,
					conv_channels[1],
					conv_kernels[1],
					conv_strides[1],
					self.drop_prob,
					name="classify_2"
				)
			self._cancer_weights.append((weights, bias))
			self._cancer_outs.append(out)
			channels = conv_channels[1]

			out = tfu.pool_2d(
					out,
					2,
					2,
					"classify_3"
				)
			self._cancer_weights.append(None)
			self._cancer_outs.append(out)

			in_shape = out.get_shape()
			in_shape = in_shape[1].value * in_shape[2].value * in_shape[3].value

			out = tf.reshape(out, shape=[-1, in_shape])

			out, weights, bias = tfu.fc_drop_bn_relu(
				out, 
				in_shape, 
				64, 
				prob=self.drop_prob, 
				name="classify_4")

			self._cancer_weights.append((weights, bias))
			self._cancer_outs.append(out)

			
			out, weights, bias = tfu.fc_drop_bn_relu(
				out, 
				64, 
				64, 
				prob=self.drop_prob, 
				name="classify_5")

			self._cancer_weights.append((weights, bias))
			self._cancer_outs.append(out)

			out, weights, bias = tfu.fc_drop_bn_relu(
				out, 
				64, 
				1, 
				prob=self.drop_prob, 
				name="classify_6_out")

			self._cancer_weights.append((weights, bias))
			self._cancer_outs.append(out)

			self._cancer = out
		print("Created Classification part!")

	def create_cancer_classification_loss(self):
		print("Adding loss to Cacner Classification...")
		self._cancer_loss = tf.reduce_mean(
				tf.nn.weighted_cross_entropy_with_logits(
						targets=self.Y_cancer,
						logits=self._cancer,
						pos_weight=self.Y_cancer_weight,
						name="Cancer_Loss"
					)
			)

		self._cancer_gs = tf.Variable(0, trainable=False)
		self._cancer_lr = tf.train.exponential_decay(
			self.config.learning_rate, 
			self._cancer_gs, 
			50000, 
			self.config.decay_rate, 
			staircase=True)

		self._cancer_optimizer = tf.train.MomentumOptimizer(
			self._cancer_lr, 
			self.config.momentum).minimize(
				self._cancer_loss,
				global_step=self._cancer_gs)
		print("Added loss to Cacner Classification!")

	def create_inputs(self, image_size):
		print("Creating input Placeholders...")
		#input image
		self.X = tf.placeholder(dtype=tf.float32, 
			shape=[None, image_size[0], image_size[1], 1],
			name="in")

		#Outputs of the different heads
		
		#Nodule head
		self.Y_nodule = tf.placeholder(dtype=tf.float32,
			shape=[None, image_size[0], image_size[1], 1],
			name="out_nodule")
		self.Y_nodule_weight = tf.placeholder_with_default(input=1.0,
			shape=None,
			name="nodule_weight")
		
		#Cancer head
		self.Y_cancer = tf.placeholder(dtype=tf.float32,
			shape=[None, 1],
			name="out_cancer")
		self.Y_cancer_weight = tf.placeholder_with_default(input=1.0,
			shape=None,
			name="cancer_weight")

		#Boolean variables to check head and mode
		self.is_training = tf.placeholder(dtype=tf.bool,
			name="is_training")
		self.is_nodule = tf.placeholder(dtype=tf.bool,
			name="is_nodule")
		self.is_cancer = tf.placeholder(dtype=tf.bool,
			name="is_cancer")

		#Probability for dropout
		self.drop_prob = tf.placeholder_with_default(input=1.0,
			shape=None,
			name="dropout_probability")

		print("Created input placeholders!")

	def start(self, restore=False):
		self._init = tf.global_variables_initializer()
		self._sess = tf.Session()
		self._saver = tf.train.Saver()

		self._sess.run(self._init)

		if restore:
			checkpoint = tf.train.get_checkpoint_state(self.config.model_save_path)
			if checkpoint and checkpoint.model_checkpoint_path:
				tf.train.restore(self._sess, checkpoint.model_checkpoint_path)

		self._started = True

	def save_model(self):
		if not self._started:
			return

		self._saver.save(self._sess, self.config.model_save_path + 'model.model')

	def train_nodule(self, dl, epochs, positive_weight=1.0):
		if not self._started:
			return

		print("Training Nodules Head for {} epochs".format(epochs))
		for epoch in range(epochs):
			
			loss = 0.0
			step = 0
			dl.train(do_shuffle=True)

			for X, Y in dl.data_iter():
				X = np.expand_dims(X, axis=len(X.shape))
				Y = np.expand_dims(Y, axis=len(Y.shape))
				step += 1
				_, l = self._sess.run([self._nodule_optimizer, self._nodule_loss],
					feed_dict={
						self.X: X,
						self.Y_nodule: Y,
						self.Y_nodule_weight: positive_weight,
						self.is_training: True,
						self.is_nodule: True,
						self.is_cancer: False,
						self.drop_prob: 0.5
					})
				print("Batch Training Loss: {}".format(l))
				loss += l

			print("Epoch Training Loss: {}".format(loss/step))

			loss = 0.0
			step = 0
			dl.validate()
			for X, Y in dl.data_iter():
				X = np.expand_dims(X, axis=len(X.shape))
				Y = np.expand_dims(Y, axis=len(Y.shape))
				step += 1
				l = self._sess.run([self._nodule_loss],
					feed_dict={
						self.X: X,
						self.Y_nodule: Y,
						self.Y_nodule_weight: positive_weight,
						self.is_training: False,
						self.is_nodule: True,
						self.is_cancer: False,
						self.drop_prob: 1.0
					})
				print("Batch Validation Loss: {}".format(l[0]))
				loss += l[0]
			print("Epoch Validation Loss: {}".format(loss/step))

	def infer_nodule(self, X):
		if not self._started:
			return None

		X = np.expand_dims(X, axis=len(X.shape))
		Y_inferred = self._sess.run([self._nodule_padded],
			feed_dict={
				self.X: X,
				self.is_training: False,
				self.is_nodule: True,
				self.is_cancer: False,
				self.drop_prob: 1.0
			})

		return Y_inferred[0]
		
	def train_cancer(self, dl, epochs, positive_weight=1.0):
		if not self._started:
			return

		print("Training Cencer Classification for {} epochs".format(epochs))
		for epoch in range(epochs):
			
			loss = 0.0
			step = 0
			dl.train(do_shuffle=True)

			for X, Y in dl.data_iter():
				X = np.expand_dims(X, axis=len(X.shape))
				Y = np.expand_dims(Y, axis=len(Y.shape))
				step += 1
				_, l = self._sess.run([self._cancer_optimizer, self._cancer_loss],
					feed_dict={
						self.X: X,
						self.Y_cancer: Y,
						self.Y_cancer_weight: positive_weight,
						self.is_training: True,
						self.is_nodule: False,
						self.is_cancer: True,
						self.drop_prob: 0.5
					})
				print("Batch Training Loss: {}".format(l))
				loss += l

			print("Epoch Training Loss: {}".format(loss/step))

			loss = 0.0
			step = 0
			dl.validate()
			for X, Y in dl.data_iter():
				X = np.expand_dims(X, axis=len(X.shape))
				Y = np.expand_dims(Y, axis=len(Y.shape))
				step += 1
				l = self._sess.run([self._cancer_loss],
					feed_dict={
						self.X: X,
						self.Y_cancer: Y,
						self.Y_cancer_weight: positive_weight,
						self.is_training: False,
						self.is_nodule: False,
						self.is_cancer: True,
						self.drop_prob: 1.0
					})
				print("Batch Validation Loss: {}".format(l[0]))
				loss += l[0]
			print("Epoch Validation Loss: {}".format(loss/step))

	def infer_cancer(self, X):
		if not self._started:
			return None

		X = np.expand_dims(X, axis=len(X.shape))
		Y_inferred = sess.run([self._cancer],
						feed_dict={
							self.X: X,
							self.is_training: False,
							self.is_nodule: False,
							self.is_cancer: True,
							self.drop_prob: 1.0
					})

		return Y_inferred[0]

def get_model(config):
	return MultiHeadUnet_2D(config)