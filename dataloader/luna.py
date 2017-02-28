import numpy as np
import pandas as pd
import pickle as p
import os
import math
from dataloader.base_dataloader import BaseDataLoader

import utils.luna16_processesor as dp

class Luna16(BaseDataLoader):
	def __init__(self, config):
		super(Luna16, self).__init__(config)
		self._load()

	def data_iter(self):
		#a generator to go through the dataset in a loop
		pass

	def train(self, do_shuffle=True):
		if do_shuffle:
			self.shuffle()
		#Go into training mode
		pass

	def validate(self):
		#Go into Validation mode
		pass

	def test(self):
		#Go into test mode
		pass

	def shuffle(self):
		#Shuffle the dataset
		pass

	def _load(self):
		pass

def get_data_loader(config):
	return Luna16(config)