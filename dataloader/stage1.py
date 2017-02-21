import numpy as np
import pandas as pd
import pickle as p
import os
import math

import utils.dicom_processor as dp

class Stage1Kaggle:
	def _load_sets(self):
		print("Loading datasets")

		train_patients = pd.read_csv(self._directory + "stage1_labels.csv")
		test_patients = pd.read_csv(self._directory + "stage1_sample_submission.csv")

		for idx, row in test_patients.iterrows():
			self._test_set.append(row['id'])

		for idx, row in train_patients.iterrows():
			self._train_set.append([row['id'], row['cancer']])

		#Create permutation for random loading
		self.shuffle()

		print("Loading datasets: Done!")

	def shuffle(self):
		self._train_set = [self._train_set[i] for i in np.random.permutation(len(self._train_set))]

	def _pre_processed_exists(self):
		if not(os.path.exists(self._target_directory) 
			and os.path.isdir(self._target_directory)):
			return False

		#Check if all patients exists
		for patient in self._train_set:
			if not os.path.exists(os.path.join(self._target_directory, patient[0] + ".pick")):
				return False

		for patient in self._test_set:
			if not os.path.exists(os.path.join(self._target_directory, patient + ".pick")):
				return False

		print("Found pre-processed datasets")
		return True

	def _pre_process(self):
		if self._pre_processed_exists():
			return

		print("No pre-processed dataset found, pre-processing")
		if not(os.path.exists(self._target_directory)):
			os.makedirs(self._target_directory)

		for patient in self._train_set:
			print("Pre-processing patient: ", patient[0])
			rescaled_image = dp.get_resized(os.path.join(self._directory + patient[0]), self._size)
			#Save the rescaled image to target directory
			p.dump(rescaled_image, open(os.path.join(self._target_directory, patient[0] + ".pick"), "wb"), protocol=2)

		for patient in self._test_set:
			print("Pre-processing patient: ", patient)
			rescaled_image = dp.get_resized(os.path.join(self._directory + patient), self._size)
			p.dump(rescaled_image, open(os.path.join(self._target_directory, patient + ".pick"), "wb"), protocol=2)

		print("Pre-processing: Done!")

	def train(self):
		train_size = math.ceil((1.0 - self._val) * len(self._train_set))
		self._current_set_x = [s[0] for s in self._train_set[:train_size]]
		self._current_set_y = [s[1] for s in self._train_set[:train_size]]

		self._current_set_size = train_size

	def validate(self):
		train_size = math.ceil((1.0 - self._val) * len(self._train_set))
		self._current_set_x = [s[0] for s in self._train_set[train_size:]]
		self._current_set_y = [s[1] for s in self._train_set[train_size:]]

		self._current_set_size = len(self._current_set_x)

	def test(self):
		self._current_set_x = self._test_set[:]
		self._current_set_size = len(self._current_set_x)

		self._current_set_y = [0] * self._current_set_size

	def _load_patient(self, patient):
		return p.load(open(os.path.join(self._target_directory, patient + ".pick"), "rb"))

	def get_next(self):
		self._current_pointer = 0

		while self._current_pointer < self._current_set_size:
			batch_x = self._current_set_x[self._current_pointer: self._current_pointer+self._batch_size]
			batch_y = self._current_set_y[self._current_pointer: self._current_pointer+self._batch_size]

			self._current_pointer += self._batch_size

			yield np.stack([self._load_patient(s) for s in batch_x]), batch_y

	def _set_directories(self):
		self._directory = "data/stage1/"
		self._target_directory = "data/preprocessed/stage1/" + \
				str(self._size[0]) + "_" + str(self._size[1]) + "_" + str(self._size[2])

	def load(self, config):
		self._size = config.size
		self._padded = config.padded_images
		self._batch_size = config.batch
		self._no_val = config.no_validation
		if self._no_val:
			self._val = 0
		else:
			self._val = config.validation_ratio

		self._train_set = []
		self._test_set = []

		self._current_set_x = None
		self._current_set_y = None
		self._current_pointer = 0
		self._current_set_size = 0

		self._set_directories()
		self._load_sets()
		self._pre_process()

		self.train()

def get_data_loader():
	return Stage1Kaggle()