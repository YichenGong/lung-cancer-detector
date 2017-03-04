import utils.dicom_processor as dp
import matplotlib.pyplot as plt

from skimage import measure
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def plot_3D(img, threshold=-400):
	verts, faces = measure.marching_cubes(img, threshold)

	fig = plt.figure(figsize=(10, 10))
	ax = fig.add_subplot(111, projection='3d')

	mesh = Poly3DCollection(verts[faces], alpha=0.1)
	face_color = [0.5, 0.5, 1]
	mesh.set_facecolor(face_color)
	ax.add_collection3d(mesh)

	ax.set_xlim(0, img.shape[0])
	ax.set_ylim(0, img.shape[1])
	ax.set_zlim(0, img.shape[2])

	plt.show()

def plot_2d(img):
	plt.imshow(img, cmap=plt.cm.gray)
	plt.show()