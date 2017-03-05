import numpy as np
import pandas as pd
import pickle as p
import os
import math
from dataloader.base_dataloader import BaseDataLoader
import cv2 as cv

import utils.luna16_processor as lp
import utils.dicom_processor as dp

class Luna16(BaseDataLoader):
	def __init__(self, config):
		super(Luna16, self).__init__(config)
		self._load()

	def _draw_nodule_mask(self, empty_array, idstr, slice_id):
		if idstr in self._Y:
			if slice_id in self._Y[idstr]:
				circles = self._Y[idstr][slice_id]

				for circle in circles:
					cv.circle(empty_array, circle[0], int(round(circle[1])), 255, -1)

		return empty_array

	def data_iter(self):
		self._current_pointer = 0

		batch_X, batch_Y = [], []
		count = 0
		
		while self._current_pointer < self._current_set_size:
			img, o, s = p.load(open(os.path.join(self._target_directory, 
				self._X[self._current_pointer] + ".pick"), "rb"))
			
			for sliceIdx in range(img.shape[0]):
				batch_X.append(img[sliceIdx])
				batch_Y.append(self._draw_nodule_mask(np.zeros_like(img[sliceIdx]), 
					self._X[self._current_pointer], sliceIdx))
				count += 1

				if count % self._batch_size == 0:
					yield np.array(batch_X), np.array(batch_Y)
					count = 0
					batch_X, batch_Y = [], []

			self._current_pointer += 1

		if len(batch_X) > 0:
			yield np.array(batch_X), np.array(batch_Y)

	def train(self, do_shuffle=True):
		if do_shuffle:
			self.shuffle()
		
		train_size = int(math.ceil((1.0 - self._val) * len(self._X)))
		self._current_set_x = self._X[:train_size]
		self._current_set_size = train_size

	def validate(self):
		train_size = int(math.ceil((1.0 - self._val) * len(self._X)))
		self._current_set_x = self._X[train_size:]
		self._current_set_size = len(self._current_set_x)

	def test(self):
		#Go into test mode
		#No test mode here as we just want to train with this
		#dataset
		self.train()

	def shuffle(self):
		self._X = [self._X[i] for i in np.random.permutation(len(self._X))]

	def _get_directory(self):
		return "luna16"

	def _set_directories(self):
		self._directory = "data/" + self._get_directory()
		if self._original_size:
			self._target_directory = "data/preprocessed/" + self._get_directory() + "/original"
		else:
			self._target_directory = "data/preprocessed/" + self._get_directory() + "/" \
					+ str(self._size[0]) + "_" + str(self._size[1]) + "_" + str(self._size[2])

		self._CSVs = os.path.join(self._directory, "CSVFILES")
		self._subsets = []
		for x in range(10):
			self._subsets.append(os.path.join(self._directory, "subset" + str(x)))

	def _extract_id(self, idstr):
		return idstr[idstr.rfind('.')+1:]

	def _file_name(self, idstr):
		return "1.3.6.1.4.1.14519.5.2.1.6279.6001." + idstr + ".mhd"

	def _load_data(self):
		annotations = pd.read_csv(os.path.join(self._CSVs, "annotations.csv"))
		
		self._annotations = []
		for idx, row in annotations.iterrows():
			self._annotations.append((self._extract_id(row['seriesuid']), row['coordX'], row['coordY'], row['coordZ'], row['diameter_mm']))

		self._all_series = []
		for directory in self._subsets:
			for file in os.listdir(directory):
				idstr, ext = os.path.splitext(file)
				if ext == ".mhd":
					self._all_series.append((directory, self._extract_id(idstr)))

	def _add_to_normalize(self, image):
		mean = np.mean(image)
		std = np.std(image)
		self._mean = ((self._mean * self._count) + mean)/(self._count + 1)
		self._count += 1

		if std > self._std:
			self._std = std

	def _pre_process(self, patient):
		img, origins, spacings = lp.load_itk_image(os.path.join(patient[0], self._file_name(patient[1])))
		self._add_to_normalize(img)
		return (img, origins, spacings)

	def _load_norm_parameters(self):
		self._mean, self._std = p.load(open(os.path.join(self._target_directory, "norm_parameters.pick"), "rb"))

	def _pre_process_all(self):
		if self._pre_processed_exists():
			self._load_norm_parameters()
			print("Mean = ", self._mean, ", STD = ", self._std)
			return

		print("No pre-processed dataset found, pre-processing now...")
		if not(os.path.exists(self._target_directory)):
			os.makedirs(self._target_directory)

		size = len(self._all_series)
		for idx, patient in enumerate(self._all_series):
			print(patient[1], str(idx+1) + "/" + str(size))
			p.dump(self._pre_process(patient), open(os.path.join(self._target_directory, patient[1] + ".pick"), "wb"), protocol=2)

		print("Mean = ", self._mean, ", STD = ", self._std)
		p.dump((self._mean, self._std), open(os.path.join(self._target_directory, "norm_parameters.pick"), "wb"), protocol=2)

		print("Pre-processing Done!")

	def _pre_processed_exists(self):
		if not(os.path.exists(self._target_directory) 
			and os.path.isdir(self._target_directory)):
			return False

		#Check if all patients exists
		for patient in self._all_series:
			if not os.path.exists(os.path.join(self._target_directory, patient[1] + ".pick")):
				return False

		print("Found pre-processed datasets")
		return True

	def _construct_mask_values(self, ids):
		mask_vals = {}
		print("Creating Mask Values...")

		size = len(self._annotations)
		for idx, annotation in enumerate(self._annotations):
			print(str(idx) + "/" + str(size))
			series, z, y, x, d = annotation #Order as given in the tutorial for LUNA-16
			r = d/2.0
			try:
				img, o, s = p.load(open(os.path.join(self._target_directory, series + ".pick"), "rb"))
			except:
				continue
			if series not in mask_vals:
				mask_vals[series] = {}
			voxelCenter = dp.world_to_voxel_coord(np.array([x, y, z]), o, s)
			x, y, z = voxelCenter
			y = int(y)
			z = int(z)
			sliceRange = range(max(0, int(x - r/s[0])), int(x + r/s[0] + 1)) #1 so that we don't loose any information
			for sliceIdx in sliceRange:
				center = (z, y)
				radius = math.sqrt(max(0, r*r - ((s[0] * math.fabs(x - sliceIdx))**2)))
				if sliceIdx not in mask_vals[series]:
					mask_vals[series][sliceIdx] = []
				mask_vals[series][sliceIdx].append((center, max(radius/s[1], radius/s[2])))
		
		print("Mask values created!")
		return mask_vals

	def _load_datasets(self):
		if os.path.exists(os.path.join(self._target_directory, "nodule_info.pick")):
			self._X, self._Y = p.load(open(os.path.join(self._target_directory, "nodule_info.pick"), "rb"))
			return
		
		for patient in self._all_series:
			self._X.append(patient[1])
		
		self._Y = self._construct_mask_values(self._X)

		p.dump((self._X, self._Y), open(os.path.join(self._target_directory, "nodule_info.pick"), "wb"))

	def _load(self):
		self._voxel_width = 65 
		self._mean = 0
		self._std = 0
		self._count = 0
		self._size = 512 #Does not work in size config
		#Only works with original size
		self._original_size = True #As all the images are already of the same size
		self._padded = self._config.padded_images
		self._batch_size = self._config.batch
		self._no_val = self._config.no_validation
		if self._no_val:
			self._val = 0
		else:
			self._val = self._config.validation_ratio

		self._X = []
		self._Y = []

		self._current_set_x = None
		self._current_pointer = 0
		self._current_set_size = 0

		self._set_directories()
		self._load_data()
		self._pre_process_all()
		self._load_datasets()

		self.train()

def get_data_loader(config):
	return Luna16(config)
