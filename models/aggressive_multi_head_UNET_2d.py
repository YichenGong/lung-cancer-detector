from __future__ import print_function
import tensorflow as tf

class MultiHeadUnet_2D:
	def __init__(self, image_size=(512, 512)):
		self.create_inputs(image_size)
		self.build_encoder()

		self.create_nodule_segment_head()

		self.create_cancer_classification_head()

	def build_encoder(self):
		print("Creating encoder part...")
	
		#TODO make this into a simple loop
		#contains list of (weights, bias, output) generally
		#may change according to the type of layer	
		
		#First Level
		with tf.name_scope("Encode_Level_1"):
			self._encode_conv1_weights = tf.Variable(tf.truncated_normal(
														shape=[3, 3, 1, 64], 
														mean=0.0,
														stddev=1.0), 
													name="encode_conv1_weights")
			self._encode_conv1_bias = tf.Variable(tf.constant(
													shape=[64], 
													value=0.1), 
												name="encode_conv1_bias")
			self._encode_conv1_layer = tf.nn.conv2d(input=self.X, 
				filter=self._encode_conv1_weights,
				strides=[1, 1, 1, 1],
				padding='VALID',
				name="encode_conv1_layer")
			self._encode_conv1_out = tf.nn.relu(self._encode_conv1_layer + self._encode_conv1_bias)
			
			self._encode_conv2_weights = tf.Variable(tf.truncated_normal(shape=[3, 3, 64, 64], 
				mean=0.0,
				stddev=1.0), name="encode_conv2_weights")
			self._encode_conv2_bias = tf.Variable(tf.constant(shape=[64], 
				value=0.1), name="encode_conv2_bias")
			self._encode_conv2_layer = tf.nn.conv2d(input=self._encode_conv1_out, 
				filter=self._encode_conv2_weights,
				strides=[1, 1, 1, 1],
				padding='VALID',
				name="encode_conv1_layer")
			self._encode_conv2_out = tf.nn.relu(self._encode_conv2_layer + self._encode_conv2_bias)

			self._encode_l1_pool = tf.nn.max_pool(self._encode_conv2_out, 
										ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
										padding='VALID',
	                         			name="encode_l1_pool")

		with tf.name_scope("Encode_Level_2"):
			self._encode_conv3_weights = tf.Variable(tf.truncated_normal(
														shape=[3, 3, 64, 128], 
														mean=0.0,
														stddev=1.0), 
													name="encode_conv3_weights")
			self._encode_conv3_bias = tf.Variable(tf.constant(
													shape=[128], 
													value=0.1), 
												name="encode_conv3_bias")
			self._encode_conv3_layer = tf.nn.conv2d(input=self._encode_l1_pool, 
				filter=self._encode_conv3_weights,
				strides=[1, 1, 1, 1],
				padding='VALID',
				name="encode_conv3_layer")
			self._encode_conv3_out = tf.nn.relu(self._encode_conv3_layer + self._encode_conv3_bias)
			
			self._encode_conv4_weights = tf.Variable(tf.truncated_normal(shape=[3, 3, 128, 128], 
				mean=0.0,
				stddev=1.0), name="encode_conv4_weights")
			self._encode_conv4_bias = tf.Variable(tf.constant(shape=[128], 
				value=0.1), name="encode_conv4_bias")
			self._encode_conv4_layer = tf.nn.conv2d(input=self._encode_conv3_out, 
				filter=self._encode_conv4_weights,
				strides=[1, 1, 1, 1],
				padding='VALID',
				name="encode_conv4_layer")
			self._encode_conv4_out = tf.nn.relu(self._encode_conv4_layer + self._encode_conv4_bias)

			self._encode_l2_pool = tf.nn.max_pool(self._encode_conv4_out, 
										ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
										padding='VALID',
	                         			name="encode_l2_pool")

		with tf.name_scope("Encode_Level_3"):
			self._encode_conv5_weights = tf.Variable(tf.truncated_normal(
														shape=[3, 3, 128, 256], 
														mean=0.0,
														stddev=1.0), 
													name="encode_conv5_weights")
			self._encode_conv5_bias = tf.Variable(tf.constant(
													shape=[256], 
													value=0.1), 
												name="encode_conv5_bias")
			self._encode_conv5_layer = tf.nn.conv2d(input=self._encode_l2_pool, 
				filter=self._encode_conv5_weights,
				strides=[1, 1, 1, 1],
				padding='VALID',
				name="encode_conv5_layer")
			self._encode_conv5_out = tf.nn.relu(self._encode_conv5_layer + self._encode_conv5_bias)
			
			self._encode_conv6_weights = tf.Variable(tf.truncated_normal(shape=[3, 3, 256, 256], 
				mean=0.0,
				stddev=1.0), name="encode_conv6_weights")
			self._encode_conv6_bias = tf.Variable(tf.constant(shape=[256], 
				value=0.1), name="encode_conv6_bias")
			self._encode_conv6_layer = tf.nn.conv2d(input=self._encode_conv5_out, 
				filter=self._encode_conv6_weights,
				strides=[1, 1, 1, 1],
				padding='VALID',
				name="encode_conv6_layer")
			self._encode_conv6_out = tf.nn.relu(self._encode_conv6_layer + self._encode_conv6_bias)

			self._encode_l3_pool = tf.nn.max_pool(self._encode_conv6_out, 
										ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
										padding='VALID',
	                         			name="encode_l3_pool")

		with tf.name_scope("Encode_Level_4"):
			self._encode_conv7_weights = tf.Variable(tf.truncated_normal(
														shape=[3, 3, 256, 512], 
														mean=0.0,
														stddev=1.0), 
													name="encode_conv7_weights")
			self._encode_conv7_bias = tf.Variable(tf.constant(
													shape=[512], 
													value=0.1), 
												name="encode_conv7_bias")
			self._encode_conv7_layer = tf.nn.conv2d(input=self._encode_l3_pool, 
				filter=self._encode_conv7_weights,
				strides=[1, 1, 1, 1],
				padding='VALID',
				name="encode_conv7_layer")
			self._encode_conv7_out = tf.nn.relu(self._encode_conv7_layer + self._encode_conv7_bias)
			
			self._encode_conv8_weights = tf.Variable(tf.truncated_normal(shape=[3, 3, 512, 512], 
				mean=0.0,
				stddev=1.0), name="encode_conv8_weights")
			self._encode_conv8_bias = tf.Variable(tf.constant(shape=[512], 
				value=0.1), name="encode_conv8_bias")
			self._encode_conv8_layer = tf.nn.conv2d(input=self._encode_conv7_out, 
				filter=self._encode_conv8_weights,
				strides=[1, 1, 1, 1],
				padding='VALID',
				name="encode_conv8_layer")
			self._encode_conv8_out = tf.nn.relu(self._encode_conv8_layer + self._encode_conv8_bias)

			self._encode_l4_pool = tf.nn.max_pool(self._encode_conv8_out, 
										ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
										padding='VALID',
	                         			name="encode_l4_pool")

		with tf.name_scope("Encode_Level_5"):
			self._encode_conv9_weights = tf.Variable(tf.truncated_normal(
														shape=[3, 3, 512, 1024], 
														mean=0.0,
														stddev=1.0), 
													name="encode_conv9_weights")
			self._encode_conv9_bias = tf.Variable(tf.constant(
													shape=[1024], 
													value=0.1), 
												name="encode_conv9_bias")
			self._encode_conv9_layer = tf.nn.conv2d(input=self._encode_l4_pool, 
				filter=self._encode_conv9_weights,
				strides=[1, 1, 1, 1],
				padding='VALID',
				name="encode_conv9_layer")
			self._encode_conv9_out = tf.nn.relu(self._encode_conv9_layer + self._encode_conv9_bias)
			
			self._encode_conv10_weights = tf.Variable(tf.truncated_normal(shape=[3, 3, 1024, 1024], 
				mean=0.0,
				stddev=1.0), name="encode_conv10_weights")
			self._encode_conv10_bias = tf.Variable(tf.constant(shape=[1024], 
				value=0.1), name="encode_conv10_bias")
			self._encode_conv10_layer = tf.nn.conv2d(input=self._encode_conv9_out, 
				filter=self._encode_conv10_weights,
				strides=[1, 1, 1, 1],
				padding='VALID',
				name="encode_conv1_layer")
			self._encode = self._encode_conv10_out = tf.nn.relu(self._encode_conv10_layer + self._encode_conv10_bias)

		print("Created encoder part!")

	def create_nodule_segment_head(self):
		'''
		We'll have more aggressive Deconvolution
		as compared to what is given in the U-NET

		Two 4x4 Deconvolutions with skip connections
		and last layer of 1x1 Convolution
		'''
		print("Creating Nodule Segmentation part...")
		with tf.name_scope("Nodule_Segment_Level_1"):
			self._nodule_upconv1_weights = tf.Variable(tf.truncated_normal(
														shape=[4, 4, 1024, 256], 
														mean=0.0,
														stddev=1.0), 
													name="nodule_upconv1_weights")
			self._nodule_upconv1_bias = tf.Variable(tf.truncated_normal(
				shape=[256],
				mean=0.0,
				stddev=1.0),
			name="nodule_upconv1_bias")
			inp_shape = self._encode.get_shape()
			self._nodule_upconv1_layer = tf.nn.conv2d_transpose(value=self._encode,
				filter=self._nodule_upconv1_weights,
				output_shape=[inp_shape[0], inp_shape[1]*4, inp_shape[2]*4, inp_shape[3]//4],
				strides=[1, 4, 4, 1],
				padding='VALID')
			self._nodule_upconv1_out = tf.nn.relu(self._nodule_upconv1_layer + self._nodule_upconv1_bias)

			x1_shape = tf.shape(self._encode_l2_pool)
			x2_shape = tf.shape(self._nodule_upconv1_out)
			# offsets for the top left corner of the crop
			offsets = [0, (x1_shape[1] - x2_shape[1]) // 2, (x1_shape[2] - x2_shape[2]) // 2, 0]
			size = [-1, x2_shape[1], x2_shape[2], -1]
			x1_crop = tf.slice(self._encode_l2_pool, offsets, size)
			self._nodule_upconv1_concat = tf.concat([x1_crop, self._nodule_upconv1_out], 3)

			self._nodule_conv11_weights = tf.Variable(tf.truncated_normal(
														shape=[3, 3, 512, 256], 
														mean=0.0,
														stddev=1.0), 
													name="nodule_conv11_weights")
			self._nodule_conv11_bias = tf.Variable(tf.truncated_normal(
				shape=[256],
				mean=0.0,
				stddev=1.0),
			name="nodule_conv11_bias")
			self._nodule_conv11_layer = tf.nn.conv2d(input=self._nodule_upconv1_concat,
				filter=self._nodule_conv11_weights,
				strides=[1, 1, 1, 1],
				padding='VALID',
				name="nodule_conv11_layer")
			self._nodule_conv11_out = tf.nn.relu(self._nodule_conv11_layer + self._nodule_conv11_bias)

		with tf.name_scope("Nodule_Segment_Level_2"):
			self._nodule_upconv2_weights = tf.Variable(tf.truncated_normal(
														shape=[4, 4, 256, 64], 
														mean=0.0,
														stddev=1.0), 
													name="nodule_upconv2_weights")
			self._nodule_upconv2_bias = tf.Variable(tf.truncated_normal(
				shape=[64],
				mean=0.0,
				stddev=1.0),
			name="nodule_upconv2_bias")
			inp_shape = self._nodule_conv11_out.get_shape()
			self._nodule_upconv2_layer = tf.nn.conv2d_transpose(value=self._nodule_conv11_out,
				filter=self._nodule_upconv2_weights,
				output_shape=[inp_shape[0], inp_shape[1]*4, inp_shape[2]*4, inp_shape[3]//4],
				strides=[1, 4, 4, 1],
				padding='VALID')
			self._nodule_upconv2_out = tf.nn.relu(self._nodule_upconv2_layer + self._nodule_upconv2_bias)

			x1_shape = tf.shape(self._encode_conv2_out)
			x2_shape = tf.shape(self._nodule_upconv2_out)
			# offsets for the top left corner of the crop
			offsets = [0, (x1_shape[1] - x2_shape[1]) // 2, (x1_shape[2] - x2_shape[2]) // 2, 0]
			size = [-1, x2_shape[1], x2_shape[2], -1]
			x1_crop = tf.slice(self._encode_conv2_out, offsets, size)
			self._nodule_upconv2_concat = tf.concat([x1_crop, self._nodule_upconv2_out], 3)

			self._nodule_conv12_weights = tf.Variable(tf.truncated_normal(
														shape=[3, 3, 128, 64], 
														mean=0.0,
														stddev=1.0), 
													name="nodule_conv12_weights")
			self._nodule_conv12_bias = tf.Variable(tf.truncated_normal(
				shape=[64],
				mean=0.0,
				stddev=1.0),
			name="nodule_conv12_bias")
			self._nodule_conv12_layer = tf.nn.conv2d(input=self._nodule_upconv2_concat,
				filter=self._nodule_conv12_weights,
				strides=[1, 1, 1, 1],
				padding='VALID',
				name="nodule_conv12_layer")
			self._nodule_conv12_out = tf.nn.relu(self._nodule_conv12_layer + self._nodule_conv12_bias)

		with tf.name_scope("Nodule_Segment_Out"):
			self._nodule_conv13_weights = tf.Variable(tf.truncated_normal(
														shape=[1, 1, 64, 1], 
														mean=0.0,
														stddev=1.0), 
													name="nodule_conv13_weights")
			self._nodule_conv13_bias = tf.Variable(tf.truncated_normal(
				shape=[1],
				mean=0.0,
				stddev=1.0),
			name="nodule_conv13_bias")
			self._nodule_conv13_layer = tf.nn.conv2d(input=self._nodule_conv12_out,
				filter=self._nodule_conv13_weights,
				strides=[1, 1, 1, 1],
				padding='VALID',
				name="nodule_conv13_layer")
			self._nodule = self._nodule_conv13_out = tf.nn.relu(self._nodule_conv13_layer + self._nodule_conv13_bias)

		print("Created Nodule Segmentation part!")

	def create_cancer_classification_head(self):
		'''
		Cancer classification head
		This will be a simple classification head 
		of cancer or not
		'''
		pass

	def create_inputs(self, image_size):
		print("Creating input Placeholders...")
		#input image
		self.X = tf.placeholder(dtype=tf.float32, 
			shape=[None, image_size[0], image_size[1], 1],
			name="in")

		#Outputs of the different heads
		self.Y_nodule = tf.placeholder(dtype=tf.float32,
			shape=[None, image_size[0], image_size[1], 1],
			name="out_nodule")
		self.Y_cancer = tf.placeholder(dtype=tf.float32,
			shape=[None, 1],
			name="out_cancer")

		#Boolean variables to check head and mode
		self.is_training = tf.placeholder(dtype=tf.bool,
			shape=[1],
			name="is_training")
		self.is_nodule = tf.placeholder(dtype=tf.bool,
			shape=[1],
			name="is_nodule")
		self.is_cancer = tf.placeholder(dtype=tf.bool,
			shape=[1],
			name="is_cancer")

		print("Created input placeholders!")