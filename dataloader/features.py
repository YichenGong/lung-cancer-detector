import cPickle
import random

import numpy as np
import pandas as pd

from dataloader.base_dataloader import BaseDataLoader


class FeatureDataLoader(BaseDataLoader):
  def __init__(self, config):
    super(FeatureDataLoader, self).__init__(config)

    self.resize_to = config.size[0]
    self.batch_size = config.batch
    self.validation_rate =config.validation_ratio
    self.k = config.top_k
    # set data dir and file
    self.data_dir = 'data/'
    self.pkl_dir = self.data_dir + 'pkl/'
    self.data_file_name = 'kaggle_stage1.pkl'
    self.feature_file_name = 'features_detect_kaggle.pkl'

    self.train_ids, self.valid_ids, self.test_ids = self.split_dataset()
    # self.train_ids, self.valid_ids, self.test_ids  = self.get_ids_from_sample_dataset()
    self.current_ids = self.train_ids
    self.current_pointer = 0
    self.feature_layers = ['46','47','56','57', '66','67','76','77']

    self.data = self.build_data_dict(self.feature_layers)
    self.feats = {}



  def data_iter(self):
      x = np.array([self.get_all_features(id, self.feature_layers) for id in self.current_ids])
      y = np.array([self.data[id]['label'] for id in self.current_ids])

      return x, y, self.current_ids


  def train(self, do_shuffle=True):
    self.current_ids = self.train_ids
    self.reset()
    self.shuffle()

  def validate(self):
    self.current_ids = self.valid_ids
    self.reset()

  def test(self):
    self.current_ids = self.test_ids
    self.reset()

  def shuffle(self):
    self.current_ids = [self.current_ids[i] for i in np.random.permutation(len(self.current_ids))]

  def reset(self):
    self.current_pointer = 0


  def get_all_features(self, pid,layer_features, k=5):
    if pid not in self.feats:
      d = self.data[pid]
      feats = []
      feats.extend(d['spacing'])
      # 0,1,2

      for i in range(k):
        feats.extend(d['p_{}'.format(i)])
      # 3,..., 7

      for i in range(k):
        feats.extend(d['loc_{}'.format(i)])
      # 8, ..., 22

      for i in range(k):
        for layer in layer_features:
          feats.extend(d['out_{}_{}'.format(i, layer)])
      self.feats[pid] = feats
    return self.feats[pid]



  def split_dataset(self):
    ids = pd.read_csv(self.data_dir + "stage1_labels.csv").id.tolist()
    random.shuffle(ids)
    split_point = int(len(ids) * self.validation_rate)

    train_ids = ids[split_point:]
    valid_ids = ids[:split_point]
    test_ids = pd.read_csv(self.data_dir + "stage1_sample_submission.csv").id.tolist()

    return train_ids, valid_ids, test_ids



  def build_data_dict(self, layer_features, k = 5):
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
        data_dict[pid]['loc_{}'.format(i)] = f['loc_{}'.format(i)]
        data_dict[pid]['p_{}'.format(i)] = f['p_{}'.format(i)]
        for layer in layer_features:
          data_dict[pid]['out_{}_{}'.format(i, layer)] = f['out_{}_{}'.format(i, layer)]

    return data_dict




