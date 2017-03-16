import os

import numpy as np
import dicom
import cv2 as cv
import utils.image_utils as imu

import scipy.ndimage as nd

#Functions inspired by https://www.kaggle.com/gzuidhof/data-science-bowl-2017/full-preprocessing-tutorial

def load_scan(filepath):
	slices = [dicom.read_file(filepath + '/' + s) for s in os.listdir(filepath)]
	slices.sort(key=lambda x: x.ImagePositionPatient[2])

	try:
		slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
	except:
		slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)

	for s in slices:
		s.SliceThickness = slice_thickness

	return slices

def get_slices_HU(slices):
	image = (np.stack([s.pixel_array for s in slices])).astype(np.int16)

	image[image == -2000] = 0

	for i, s in enumerate(slices):
		image[i] = (s.RescaleSlope * image[i].astype(np.float64)).astype(np.int16)
		image[i] += np.int16(s.RescaleIntercept)

	return np.array(image, dtype=np.int16)

def is_scan_processable(scan):
	size = len(scan)
	if size > 0:
		typeOfScan = scan[0].SOPClassUID

		#Ignoreing CR and DX scans
		if typeOfScan == 'Digital X-Ray Image Storage - For Presentation':
			return False
		if typeOfScan == 'Computed Radiography Image Storage':
			return False
		if typeOfScan == 'Segmentation Storage':
			return False
	else:
		return False

	return True

def load_lidc_scan(filepath, resize=None, print_details=False):
	slices = [dicom.read_file(filepath + '/' + s) for s in os.listdir(filepath)]

	if not is_scan_processable(slices):
		print("Not processable: ", filepath)
		return None

	if print_details:
		print(slices[0])
		print(len(slices))
		print(filepath)
		return

	slices.sort(key=lambda x: x.ImagePositionPatient[2])

	image = get_slices_HU(slices)
	origShape = image.shape
	if resize:
		image = get_resized_image(image, resize)

	spacing = np.array(slices[0].PixelSpacing + [slices[0].SliceThickness], dtype=np.float32)
	origin = np.array(slices[0].ImagePositionPatient)

	return image, spacing, origin, origShape

def get_image_HU(filepath):
	slices = load_scan(filepath)
	return get_slices_HU(slices)

def get_resampled(filepath, new_spacing=[1, 1, 1]):
	scan = load_scan(filepath)
	image = get_image_HU(filepath)

	spacing = np.array([scan[0].SliceThickness] + scan[0].PixelSpacing, dtype=np.float32)

	resize_factor = spacing / new_spacing
	new_shape = np.round(image.shape * resize_factor)
	real_resize_factor = new_shape / image.shape

	return nd.interpolation.zoom(image, real_resize_factor, mode='nearest')

def get_resized(filepath, new_size, padding=False):
	image = get_image_HU(filepath)
	return get_resized_image(image, new_size, padding)

def get_resized_image(image, new_size, padding=False):
	return imu.resize_3d(image, new_size, padding)

def world_to_voxel_coord(worldCoord, origin, spacing):
	strectchedVoxelCoord = np.absolute(worldCoord - origin)
	voxelCoord = strectchedVoxelCoord / spacing

	return voxelCoord

def normalize_planes(npzarray, maxHU=400., minHU=-1000.):
	maxHU, minHU = float(maxHU), float(minHU)

	npzarray = (npzarray - minHU) / (maxHU - minHU)
	npzarray[npzarray > 1] = 1
	npzarray[npzarray < 0] = 0

	return npzarray