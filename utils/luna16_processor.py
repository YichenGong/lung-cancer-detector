'''
Utility functions taken from: https://luna16.grand-challenge.org/tutorial/
'''

import SimpleITK as sitk
import numpy as np

def load_itk_image(filepath):
	itkimage = sitk.ReadImage(filepath)
	img = sitk.GetArrayFromImage(itkimage)

	origins = np.array(list(reversed(itkimage.GetOrigin())))
	spacings = np.array(list(reversed(itkimage.GetSpacing())))

	return img, origins, spacings

def world_to_voxel_coord(worldCoord, origin, spacing):
	strectchedVoxelCoord = np.absolute(worldCoord - origin)
	voxelCoord = strectchedVoxelCoord / spacing

	return voxelCoord

def normalize_planes(npzarray):
	maxHU = 400
	minHU = -1000

	npzarray = (npzarray - minHU) / (maxHU - minHU)
	npzarray[npzarray > 1] = 1
	npzarray[npzarray < 0] = 0

	return npzarray