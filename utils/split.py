from __future__ import print_function
import csv
import json
import numpy as np
import math


def get_test_ids(path):
  test_ids = []
  with open(path, 'rb') as f:
    submission = csv.reader(f)
    for row in submission:
      test_ids.append(row[0])

  # skip the attribute name 'id'
  return test_ids[1:]

def get_positive_negative(path):
  positive = []
  negative = []
  with open(path, 'rb') as f:
    submission = csv.reader(f)
    for row in submission:
      if row[1] == '1':
        positive.append(row[0])
      elif row[1] == '0':
        negative.append(row[0])

  return positive, negative

def get_train_and_validation_ids(path, percent=0.2):
  positive, negative = get_positive_negative(path)
  positive = np.random.permutation(positive)
  negative = np.random.permutation(negative)

  num_val_pos = int(len(positive) * percent)
  num_val_neg = int(len(negative) * percent)

  train = []
  valid = []
  train.extend(positive[num_val_pos:].tolist())
  train.extend(negative[num_val_neg:].tolist())
  valid.extend(positive[:num_val_pos].tolist())
  valid.extend(negative[:num_val_neg].tolist())
  return train, valid


if __name__ == '__main__':
  submission_file_path = 'stage1_sample_submission.csv'
  stage1_labels_path = 'stage1_labels.csv'

  train, validation = get_train_and_validation_ids(stage1_labels_path)
  test = get_test_ids(submission_file_path)
  print(len(train))
  print(len(validation))
  print(len(test))
  print(set(train).intersection(validation))

  result = {'train':train,
            'validation':validation,
            'test': test}

  json.dump(result, open("split_backup.json", 'w'))
