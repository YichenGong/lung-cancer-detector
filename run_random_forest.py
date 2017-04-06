import csv

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss
from matplotlib import pyplot as plt
import numpy as np
import options
from dataloader.features import FeatureDataLoader


def main():
  opt = options.parse()
  opt.top_k = 5
  opt.validation_ratio = 0.2

  ##########################################
  # train
  # clf = GradientBoostingClassifier(n_estimators=3000, learning_rate=0.05, max_depth=6, max_features='sqrt')
  # clf = RandomForestClassifier()
  clf = RandomForestClassifier(n_estimators=3500,
                             criterion='entropy',
                             max_depth=None,
                             max_features='sqrt',
                             min_samples_split=4,
                             min_samples_leaf=2,
                             n_jobs=4)

  count = 0
  test_predictions = None

  for i in range(10):
    dl = FeatureDataLoader(opt)
    x_train, y_train, x_valid, y_valid, x_test, test_ids = get_data(dl)

    clf.fit(x_train, y_train)
    clf_probs = clf.predict_proba(x_valid)[:, 1]
    score = log_loss(y_valid, clf_probs)
    test_predictions = clf.predict_proba(x_test)[:, 1]

    print(score)
    # if score < 0.44:
    #   print('{}:{}'.format(i, score))
    #   if test_predictions is None:
    #     test_predictions = clf.predict_proba(x_test)[:, 1]
    #   else:
    #     test_predictions += clf.predict_proba(x_test)[:, 1]
    #   count += 1
  #
  # test_predictions  = test_predictions / count

  with open('submission_random_forest.csv', 'w') as f:
    writer = csv.writer(f)
    # write the header
    for row in {'id':'cancer'}.items():
      writer.writerow(row)
    # write the content
    for row in zip(test_ids, test_predictions):
      writer.writerow(row)



  # print(log_loss(y_train, clf.predict_proba(x_train)))

  # print(score)
  # print(normalized_score)

  test_predictions = clf.predict_proba(x_test)[:, 1]
  # print(max(test_predictions), min(test_predictions))

  ###################################
  # Check importance
  # importance = clf.feature_importances_
  # print(importance)
  # print(sorted(range(len(importance)), key=importance.__getitem__, reverse=True))

  ###################################
  # Check prediction
  # print(clf_probs[y_valid==1])
  # print(clf_probs[y_valid==0])



def normalize(data):
  return (data - min(data)) / (max(data) - min(data))

def get_data(dl, feature_idx=None):
  dl.train()
  x_train, y_train, train_ids = dl.data_iter()
  dl.validate()
  x_valid, y_valid, valid_ids = dl.data_iter()
  dl.test()
  x_test, _, test_ids = dl.data_iter()

  if feature_idx:
    x_train = x_train[:, feature_idx]
    x_valid = x_valid[:, feature_idx]
    x_test = x_test[:, feature_idx]

  return x_train, y_train, x_valid, y_valid, x_test, test_ids

if __name__ == '__main__':
  main()