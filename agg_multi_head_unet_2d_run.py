from __future__ import print_function
import os
import tensorflow as tf
import options
import models.aggressive_multi_head_UNET_2d as unet

from dataloader import lidc, luna, sample, stage1

opt = options.parse()

# if opt.restore:
# 		net = unet.get_model()

net = unet.get_model()
if opt.amhu2_luna_train:
	dl_luna = luna.get_data_loader(opt)

	net.train_nodule(dl_luna, 
		opt.epochs, 
		opt.learning_rate, 
		opt.momentum, 
		opt.decay_rate)
