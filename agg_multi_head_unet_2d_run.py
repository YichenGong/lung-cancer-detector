from __future__ import print_function
import os
import tensorflow as tf
import options
import models.aggressive_multi_head_UNET_2d as unet
import numpy as np

opt = options.parse()
print(opt)

net = unet.get_model(opt)
net.start(restore=True)
if (not opt.amhu2_luna_lidc_train) and opt.amhu2_luna_train:
	from dataloader import luna
	opt.original = True
	dl_luna = luna.get_data_loader(opt)

	net.train_nodule(dl_luna, 
		opt.epochs,
		opt.false_negative_weight)

if (not opt.amhu2_luna_lidc_train) and opt.amhu2_lidc_train:
	from dataloader import lidc
	opt.original = True
	dl_lidc = lidc.get_data_loader(opt)

	net.train_nodule(dl_lidc, 
		opt.epochs, 
		opt.false_negative_weight)

if opt.amhu2_luna_lidc_train:
	from dataloader import lidc, luna
	opt.original = True
	class DataMixer:
		def __init__(self):
			self.dl_luna = luna.get_data_loader(opt)
			self.dl_lidc = lidc.get_data_loader(opt)

		def data_iter(self):
			data_1 = self.dl_luna.data_iter()
			data_2 = self.dl_lidc.data_iter()

			flag_1, flag_2 = False, False
			while True:
				data = next(data_1, None)
				if data is not None:
					yield data
				else:
					flag_1 = True

				data = next(data_2, None)
				if data is not None:
					yield data
				else:
					flag_2 = True

				if flag_1 and flag_2:
					break

		def train(self, do_shuffle=True):
			self.dl_luna.train(do_shuffle)
			self.dl_lidc.train(do_shuffle)

		def validate(self):
			self.dl_lidc.validate()
			self.dl_luna.validate()

		def test(self):
			self.dl_lidc.test()
			self.dl_luna.test()

	dl = DataMixer()
	net.train_nodule(dl, 
		opt.epochs, 
		opt.false_negative_weight)

if opt.amhu2_nodule_cancer_train:
	from dataloader import lidc
	class NoduleCancerLayers:
		def __init__(self, dl, opt):
			self.dl = dl.get_data_loader(opt)

		def data_iter(self):
			for X, Y in self.dl.data_iter():
				Y_cancer = np.zeros(Y.shape[0])
				for idx in range(Y.shape[0]):
					if np.count_nonzero(Y[idx]) > 0:
						Y_cancer[idx] = 1
				yield X, Y_cancer

		def train(self, do_shuffle=True):
			self.dl.train(do_shuffle)

		def validate(self):
			self.dl.validate()

		def test(self):
			self.dl.test()

	dl_nodule_cancer = NoduleCancerLayers(lidc, opt)

	net.train_cancer(dl_nodule_cancer,
		opt.epochs,
		opt.false_negative_weight)


class KaggleSingleLayer:
	def __init__(self, dl, opt):
		self.batch = opt.batch
		opt.batch = 1
		self.dl = dl.get_data_loader(opt)
		opt.batch = self.batch

	def data_iter(self):
		for X, Y, Z in self.dl.data_iter():
			X = np.squeeze(X, axis=0)
			Y = np.full(shape=(X.shape[0]), fill_value=Y[0], dtype=Y.dtype)

			counter = 0
			while counter < X.shape[0]:
				batch_X = X[counter : counter + self.batch, :]
				batch_Y = Y[counter : counter + self.batch]

				yield batch_X, batch_Y
				counter += self.batch

	def train(self, do_shuffle=True):
		self.dl.train(do_shuffle)

	def validate(self):
		self.dl.validate()

	def test(self):
		self.dl.test()

if opt.amhu2_sample_train:
	from dataloader import sample
	opt.original = True
	dl_sample = KaggleSingleLayer(sample, opt)

	net.train_cancer(dl_sample,
		opt.epochs,
		opt.false_negative_weight)

if opt.amhu2_stage1_train:
	from dataloader import stage1
	opt.original = True
	dl_stage1 = KaggleSingleLayer(stage1, opt)

	net.train_cancer(dl_stage1,
		opt.epochs,
		opt.false_negative_weight)