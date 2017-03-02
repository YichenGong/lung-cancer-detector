import numpy as np
import cv2 as cv
import random
import scipy.ndimage as nd

def hflip_2d(img, random_flip=True, probability=0.5):
	if random_flip:
		if random.random() < probability:
			return img

	return cv.flip(img, 0)

def vflip_2d(img, random_flip=True, probability=0.5):
	if random_flip:
		if random.random() < probability:
			return img

	return cv.flip(img, 1)

def rotate_2d(img, angle_in_degrees,random_rotate=True, probability=0.5):
	if random_flip:
		if random.random() < probability:
			return img

	rot_mat = cv.getRotationMatrix2D(tuple(np.array(img.shape)/2), angle, 1.0)
	return cv.warpAffine(img, rot_mat, img.shape, flags=cv.INTER_LINEAR)

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