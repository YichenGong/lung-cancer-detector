import os

import numpy as np
import dicom
import cv2 as cv

import scipy.ndimage as nd
import matplotlib.pyplot as plt

from skimage import measure, morphology
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

#Functions inspired by https://www.kaggle.com/gzuidhof/data-science-bowl-2017/full-preprocessing-tutorial

def load_scan(filepath):
	slices = [dicom.read_file(filepath + '/' + s) for s in os.listdir(filepath)]
	slices.sort(key=lambda x: int(x.ImagePositionPatient[2]))

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

def load_lidc_scan(filepath, resize=None):
	slices = [dicom.read_file(filepath + '/' + s) for s in os.listdir(filepath)]
	slices.sort(key=lambda x: int(x.ImagePositionPatient[2]))

	if resize:
		image = get_slices_HU(slices)
		origShape = image.shape
		image = get_resized_image(image, resize)

	spacing = np.array([slices[0].SliceThickness] + slices[0].PixelSpacing, dtype=np.float32)
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

def get_resized(filepath, new_size):
	image = get_image_HU(filepath)
	return get_resized_image(image, new_size)

def get_resized_image(image, new_size):
	if new_size[0] == -1:
		#Resize 2D wise!
		return np.array([cv.resize(single_slice, (new_size[1], new_size[2])) \
			for single_slice in scan])
	else:
		resize_factor = [a/float(b) for a,b in zip(new_size, image.shape)]
		return nd.interpolation.zoom(image, resize_factor, mode='nearest')

def plot_3D(image, threshold=-400):
	verts, faces = measure.marching_cubes(image, threshold)

	fig = plt.figure(figsize=(10, 10))
	ax = fig.add_subplot(111, projection='3d')

	mesh = Poly3DCollection(verts[faces], alpha=0.1)
	face_color = [0.5, 0.5, 1]
	mesh.set_facecolor(face_color)
	ax.add_collection3d(mesh)

	ax.set_xlim(0, image.shape[0])
	ax.set_ylim(0, image.shape[1])
	ax.set_zlim(0, image.shape[2])

	plt.show()
