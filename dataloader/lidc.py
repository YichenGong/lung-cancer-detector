from __future__ import print_function
import numpy as np
import pandas as pd
import pickle as p
import os
import math
from dataloader.base_dataloader import BaseDataLoader

from utils import dicom_processor as dp, lidc_xml_parser

class LIDCData(BaseDataLoader):
	def __init__(self, config):
		super(LIDCData, self).__init__(config)
		self._load()

	def data_iter(self):
		#a generator to go through the dataset in a loop
		pass

	def train(self, do_shuffle=True):
		#Go into training mode
		pass

	def validate(self):
		#Go into Validation mode
		pass

	def test(self):
		#No testing, only validation
		self.validate()

	def shuffle(self):
		#Shuffle the dataset
		pass

	def _studies_directory_iter(self):
		for i in os.listdir(self._studies):
			di = os.path.join(self._studies, i)
			for j in os.listdir(di):
				dj = os.path.join(di, j)
				for k in os.listdir(dj):
					dk = os.path.join(dj, k)
					name = k[k.rfind('.')+1:]
					yield dk, name

	def _pre_process_images(self):
		print("Pre-processing images...")
		
		for path, name in self._studies_directory_iter():
			print("Processing ", name)
			for s in os.listdir(path):
				root, ext = os.path.splitext(s)
				if ext != '.dcm':
					os.remove(os.path.join(path, s))
			if self._original_size:
				resize = None
			else:
				resize = self._size

			try:
				slices = dp.load_lidc_scan(os.path.join(path), resize)
			except:
				print("Error with ", path)
				dp.load_lidc_scan(os.path.join(path), resize, print_details=True)
				if name in self._nodule_info:
					print("Nodules exist for this series")
				continue
			
			p.dump(slices, 
				open(os.path.join(self._target_directory, name + ".pick"), "wb"), 
				protocol=2)

		print("Image pre-processing complete!")

	def _pre_process_XMLs(self):
		print("Pre-processing XMLs...")
		nodule_info_list = lidc_xml_parser.load_xmls(self._xmls)

		#Create a more sensible list for iteration
		#over the dataset of nodules
		self._nodule_info = {}
		for nodule_info in nodule_info_list:
			series = nodule_info['header']['uid']
			if series not in self._nodule_info:
				self._nodule_info[series] = []

			for nodule in nodule_info['readings']:
				#We'll ignore the Non-Nodules right now
				if nodule.is_nodule():
					for roi in nodule.get_roi():
						z = roi.z
						iid = roi.image_uid
						vertices = [(edge.x, edge.y) for edge in roi.get_edges()]
						self._nodule_info[series].append((iid, z, vertices))

		p.dump(self._nodule_info, 
			open(os.path.join(self._target_directory, "nodule_info.pick"), "wb"),
			protocol=2)
		print("XMLs pre-processing completes...")

	def _pre_process_exists(self):
		if not(os.path.exists(self._target_directory) 
			and os.path.isdir(self._target_directory)):
			return False

		#Check if all patients exists
		for path, name in self._studies_directory_iter():
			if not os.path.exists(os.path.join(self._target_directory, name + ".pick")):
				return False

		if not os.path.exists(os.path.join(self._target_directory, "nodule_info.pick")):
			return False


		print("Found pre-processed datasets")
		return True

	def _load_preprocessed_data(self):
		self._nodule_info = p.load(open(os.path.join(self._target_directory, "nodule_info.pick"), "rb"))

	def _pre_process(self):
		if self._pre_process_exists():
			print("Pre-processed dataset exists!")
			self._load_preprocessed_data()
			return
		print("No pre-processed dataset found...")

		if not(os.path.exists(self._target_directory)):
			os.makedirs(self._target_directory)
		
		self._pre_process_XMLs()
		self._pre_process_images()

		print("Pre-processing Done!")

	def _get_directory(self):
		return "lidc"

	def _set_directories(self):
		self._directory = "data/" + self._get_directory()
		if self._original_size:
			self._target_directory = "data/preprocessed/" + self._get_directory() + "/original"
		else:
			self._target_directory = "data/preprocessed/" + self._get_directory() + "/" \
					+ str(self._size[0]) + "_" + str(self._size[1]) + "_" + str(self._size[2])

		self._studies = os.path.join(self._directory, "studies")
		self._xmls = os.path.join(self._directory, "XMLs")

	def _load(self):
		self._size = self._config.size
		self._original_size = self._config.original

		self._batch_size = self._config.batch
		self._no_val = self._config.no_validation
		if self._no_val:
			self._val = 0
		else:
			self._val = self._config.validation_ratio

		self._set_directories()
		self._pre_process()

		self.train()

def get_data_loader(config):
	return LIDCData(config)
