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
