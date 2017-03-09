import cPickle
import os
import random
import pickle

import numpy as np
import pandas as pd
import scipy.ndimage as nd

import utils.dicom_processor as dp
from dataloader.base_dataloader import BaseDataLoader


class CandidateDataLoader(BaseDataLoader):
  def __init__(self, config):
    super(CandidateDataLoader, self).__init__(config)

    self.resize_to = config.size[0]
    self.diameter_mm = config.diameter_mm
    self.batch_size = config.batch
    self.validation_rate =config.validation_ratio
    self.k = config.top_k
    # set data dir and file
    self.data_dir = 'data/'
    self.pkl_dir = self.data_dir + 'pkl/'
    self.stage1_dir = self.data_dir + 'stage1/'
    self.sample_dir = self.data_dir + 'sample/'
    self.patch_dir = self.data_dir + 'patch/'
    self.data_file_name = 'kaggle_stage1.pkl'
    self.feature_file_name = 'features_detect_kaggle.pkl'

    self.train_ids, self.valid_ids, self.test_ids = self.split_dataset()
    # self.train_ids, self.valid_ids, self.test_ids  = self.get_ids_from_sample_dataset()
    self.current_ids = self.train_ids
    self.current_pointer = 0

    self.data = self.build_data_dict(layer_features=['67', '77'], k=self.k)



  def data_iter(self):
    current_set_size = len(self.current_ids)

    while self.current_pointer < current_set_size:
      batch_ids = self.current_ids[self.current_pointer: self.current_pointer + self.batch_size]
      batch_x = np.array([self.get_first_k_patches(id, self.k, self.diameter_mm, self.resize_to)
                          for id in batch_ids])
      batch_x = np.swapaxes(batch_x,0,1)
      batch_y = np.array([self.data[id]['label'] for id in batch_ids])

      self.current_pointer += self.batch_size
      yield expand_last_dim(batch_x, batch_y, np.array(batch_ids))


  def train(self, do_shuffle=True):
    self.current_ids = self.train_ids
    self.reset()

  def validate(self):
    self.current_ids = self.valid_ids
    self.reset()

  def test(self):
    self.current_ids = self.test_ids
    self.reset()

  def shuffle(self):
    # Shuffle the dataset
    pass

  def reset(self):
    self.current_pointer = 0


  def get_first_k_patches(self, pid, k, diameter_mm, resize_to, reuse=True):
    """
      Gets first k patches and resize them into unified size.
    """
    file_path = '{}/{}.pkl'.format(self.patch_dir, pid)

    if reuse and os.path.exists(file_path):
      with open(file_path, 'rb') as f:
        try:
          result = pickle.load(f)
        except EOFError:
          print('EOF Error: file id: {}'.format(pid))
    else:
      d = self.data[pid]
      scan = dp.get_image_HU('{}{}'.format(self.stage1_dir, pid))
      result = []
      for i in range(k):
        raw_patch = get_patch(scan, d[i]['loc'], diameter_mm, d['spacing'])
        resize_factor = [resize_to / float(patch_shape) for patch_shape in raw_patch.shape]
        patch = nd.interpolation.zoom(raw_patch, resize_factor, mode='nearest')
        result.append(patch)
      with open(file_path, 'wb') as f:
        pickle.dump(result, f)
    return result

  def split_dataset(self):
    ids = pd.read_csv(self.data_dir + "stage1_labels.csv").id.tolist()
    random.shuffle(ids)
    split_point = int(len(ids) * self.validation_rate)

    train_ids = ids[split_point:]
    valid_ids = ids[:split_point]
    test_ids = pd.read_csv(self.data_dir + "stage1_sample_submission.csv").id.tolist()
    print(len(train_ids), len(valid_ids), len(test_ids))
    return train_ids, valid_ids, test_ids

  def get_ids_from_sample_dataset(self):
    # take sample dataset both as train and test
    train_ids = pd.read_csv(self.data_dir + "stage1_labels.csv").id.tolist()
    sample_ids = os.listdir(self.sample_dir)
    ids = [id for id in sample_ids if id in train_ids]
    return ids, ids, ids


  def build_data_dict(self, layer_features, k=5):
    """
    This build dict[id] = {label, spacing, 1={loc, p, layer1_feature, layer2_feature...}, 2={}...}

    :param layer_features: features from layer, e.g 67, 77
    :param k: number of nodule considered as inputs
    :return: a combined dictionary
    """
    with open(self.pkl_dir + self.data_file_name, 'rb') as data_file:
      data = cPickle.load(data_file)
    with open(self.pkl_dir + self.feature_file_name, 'rb') as feature_file:
      features = cPickle.load(feature_file)

    data_dict = {}
    for d,f in zip(data, features):
      pid = d['id']
      data_dict[pid] = {'label':d['label'], 'spacing':d['spacing']}

      # add the features
      for i in range(k):
        data_dict[pid][i] = {'loc': f['loc_{}'.format(i)], 'p': f['p_{}'.format(i)]}
        for layer in layer_features:
          data_dict[pid][i][layer] = f['out_{}_{}'.format(i, layer)]

    return data_dict

def get_patch(scan, loc, diameter_mm, spacing):
  """ get the patch 3d matrix with the params.

  :param scan: the 3d CT scan
  :param loc: location of the nodule in x,y,z
  :param spacing: the spacing between pixels
  :param diameter_mm: diameter of the 3d cube matrix in mm.
  :return: 3d matrix of the patch extracted
  """
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
  """
    Gets the lo and hi between two bound.
    With center and radius provided.
  """
  if center - radius <= 0:
    hi = radius * 2
    lo = 0
  elif center + radius >= upper_bound:
    hi = upper_bound - 1
    lo = hi - radius * 2
  else:
    hi = center + radius
    lo = center - radius

  return lo, hi

def expand_last_dim(*input_data):
  res = []
  for in_data in input_data:
    res.append(np.expand_dims(in_data, axis=len(in_data.shape)))
  if len(res) == 1:
    return res[0]
  else:
    return res
