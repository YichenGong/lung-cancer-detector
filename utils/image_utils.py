import numpy as np
import cv2 as cv
import random
import scipy.ndimage as nd

def hflip_2d(img, random_mode=True, probability=0.5):
	if random_mode:
		if random.random() < probability:
			return img

	return cv.flip(img, 0)

def vflip_2d(img, random_mode=True, probability=0.5):
	if random_mode:
		if random.random() < probability:
			return img

	return cv.flip(img, 1)

def rotate_2d(img, angle_in_degrees,random_mode=True, probability=0.5):
	if random_mode:
		if random.random() < probability:
			return img

	rot_mat = cv.getRotationMatrix2D(tuple(np.array(img.shape)/2), angle, 1.0)
	return cv.warpAffine(img, rot_mat, img.shape, flags=cv.INTER_LINEAR)

def translate_2d(img, dxdy, random_mode=True, probability=0.5):
	if random_mode:
		if random.random() < probability:
			return img

	dx, dy = dxdy
	shift_mat = np.array([[1, 0, dx], [0, 1, dy]])
	return cv.warpAffine(img, shift_mat, img.shape, flags=cv.INTER_LINEAR)

def shear_2d(img, dxdy, random_mode=True, probability=0.5):
	if random_mode:
		if random.random() < probability:
			return img

	dx, dy = dxdy
	shear_mat = np.array([[1, dx, 0], [dy, 1, 0]])
	return cv.warpAffine(img, shear_mat, img.shape, flags=cv.INTER_LINEAR)

def elastic_transform_2d(img, alpha, sigma, random_mode=True, probability=0.5):
	#Taken from: https://gist.github.com/chsasank/4d8f68caf01f041a6453e67fb30f8f5a
	if random_mode:
		if random.random() < probability:
			return img

	dx = nd.gaussian_filter((random_state.rand(img.shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
	dy = nd.gaussian_filter((random_state.rand(img.shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha

	x, y = np.meshgrid(np.arange(img.shape[0]), np.arange(img.shape[1]), indexing='ij')
	indices = np.reshape(x+dx, (-1, 1)), np.reshape(y+dy, (-1, 1))

	return nd.map_coordinates(img, indices, order=1).reshape(img.shape)

def resize_2d(img, new_size):
	return cv.resize(img, new_size)

def resize_3d(img, new_size):
	if new_size[0] == -1:
		#Resize 2D wise!
		return np.array([cv.resize(img[idx], (new_size[1], new_size[2])) \
			for idx in range(img.shape[0])])
	else:
		resize_factor = [a/float(b) for a,b in zip(new_size, img.shape)]
		return nd.interpolation.zoom(img, resize_factor, mode='nearest')