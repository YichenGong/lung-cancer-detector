import csv

import SimpleITK as sitk
import numpy as np
import scipy.ndimage as nd
import pickle
import os

_id = 1

def get_next_id():
  global _id
  _id += 1
  return _id

def load_itk(filename):
  itkimage = sitk.ReadImage(filename)
  image = np.transpose(sitk.GetArrayFromImage(itkimage))
  origin = np.array(itkimage.GetOrigin())
  spacing = np.array(itkimage.GetSpacing())
  return image, origin, spacing


def worldToVoxelCoord(worldCoord, origin, spacing):
  stretchedVoxelCoord = np.absolute(worldCoord - origin)
  voxelCoord = stretchedVoxelCoord / spacing
  return voxelCoord

def get_patch(scan, loc, diameter_mm, spacing):
  # spacing mm/pixel
  spacing_z, spacing_x, spacing_y = spacing

  # the units of radius below is pixel
  radius_x = int(diameter_mm / spacing_x / 2)
  radius_y = int(diameter_mm / spacing_y / 2)
  radius_z = int(diameter_mm / spacing_z / 2)

  x,y,z = loc
  z_lo, z_hi = get_lo_hi_within_bound(z, radius_z, scan.shape[0])
  x_lo, x_hi = get_lo_hi_within_bound(x, radius_x, scan.shape[1])
  y_lo, y_hi = get_lo_hi_within_bound(y, radius_y, scan.shape[2])

  return scan[z_lo:z_hi, x_lo:x_hi, y_lo:y_hi]


def get_lo_hi_within_bound(center, radius, upper_bound, lower_bound=0):
  if center - radius <= lower_bound:
    hi = radius * 2
    lo = lower_bound
  elif center + radius >= upper_bound:
    hi = upper_bound - 1
    lo = hi - radius * 2
  else:
    hi = center + radius
    lo = center - radius

  return lo, hi


def normalize(image):
  MIN_BOUND = -1000.0
  MAX_BOUND = 400.0
  image = (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
  image[image>1] = 1.
  image[image<0] = 0.
  return image


def build_data_dict(path):
  result_dict = {}
  with open(path, 'rb') as csvfile:
    for row in csv.reader(csvfile):
      if row[0] == 'seriesuid':
        continue
      else:
        coord = [float(coord) for coord in row[1:-1]]
        if row[0] in result_dict:
          result_dict[row[0]].append(coord)
        else:
          result_dict[row[0]] = [coord]


  return result_dict


def extract_patches(img_dir, data_dict, patch_dir, diameter_mm):
  for uid, locs in data_dict:
    scan, origin, spacing = load_itk(img_dir + uid + '.mhd')
    for loc_world_coord in locs:
      loc = worldToVoxelCoord(loc_world_coord, origin, spacing)
      # get and preprocess patch
      raw_patch = get_patch(scan, loc, diameter_mm, spacing)
      resize_factor = [diameter_mm / float(patch_shape) for patch_shape in raw_patch.shape]
      patch = nd.interpolation.zoom(raw_patch, resize_factor, mode='nearest')
      patch = normalize(patch)

      current_id = get_next_id()
      file_path = '{}/{}.{}.pkl'.format(patch_dir, uid, current_id)
      with open(file_path, 'wb') as f:
        pickle.dump(patch, f)

      if current_id % 100 == 0:
        print(current_id)


data_dir = 'data/luna16/'
img_dir = data_dir + '/image/'

annotations = build_data_dict(data_dir + '/CSVFILES/annotations.csv')
candidates  = build_data_dict(data_dir + '/CSVFILES/candidates.csv')

extract_patches(img_dir, annotations, data_dir + 'annotation_patch/', 30)



# path = 'data/luna16/subset0/1.3.6.1.4.1.14519.5.2.1.6279.6001.219909753224298157409438012179.mhd'
#
# img, origins, spacings = load_itk(path)
#
# print img.shape, origins, spacings