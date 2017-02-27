import dataloader.stage1 as stage1

import os
import pandas as pd
import numpy as np

class SampleKaggle(stage1.Stage1Kaggle):
	def _check_sample_exists(self, patient):
		return os.path.exists(os.path.join(self._directory, patient))

	def shuffle(self):
		self._train_set = [self._train_set[i] for i in np.random.permutation(len(self._train_set))]

	def _load_sets(self):
		print("Loading datasets")

		train_patients = pd.read_csv("data/stage1/" + "stage1_labels.csv")

		for idx, row in train_patients.iterrows():
			if self._check_sample_exists(row['id']):
				self._test_set.append(row['id'])

		for idx, row in train_patients.iterrows():
			if self._check_sample_exists(row['id']):
				self._train_set.append([row['id'], row['cancer']])

		#Create permutation for random loading
		self.shuffle()

		print("Loading datasets: Done!")

	def _set_directories(self):
		self._directory = "data/sample/"
		self._target_directory = "data/preprocessed/sample/" + \
				str(self._size[0]) + "_" + str(self._size[1]) + "_" + str(self._size[2])

def get_data_loader():
	return SampleKaggle()
