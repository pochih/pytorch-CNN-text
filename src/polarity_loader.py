# -*- coding: utf-8 -*-

from __future__ import print_function

from six.moves import cPickle as pickle
import numpy as np
import codecs
import random
import os

import torch
from torch.utils.data import Dataset
from wordvec import WordVec


class PolarityDataset(Dataset):

  def __init__(self, phase, wv_type='glove'):
    '''
      Parameters
      ----------
      phase   : phase of the task, equals to one in {'train', 'val'}
      wv_type : word vector type, equals to one in {'google', 'glove', 'self'}
    '''
    self.phase = phase
    self.data = self.get_data(self.phase)
    self.wordvec = WordVec(wv_type)

  def __len__(self):
    return len(self.data)

  def __getitem__(self, idx):
    sent = self.data[idx][0]
    sent = np.array([self._get_wv(word) for word in sent.split()]).T  # get every word's vector in given sentence
    sent = torch.from_numpy(sent).float()
    cls = self.data[idx][1]
    target = np.zeros(2)
    target[cls] = 1
    target = torch.from_numpy(target).float()
    sample = {'X': sent, 'Y': target}
    return sample

  def get_data(self, phase, val_rate=0.1):
    if not os.path.exists('rt-polaritydata/train.pkl') or not os.path.exists('rt-polaritydata/val.pkl'):
      pos = codecs.open("rt-polaritydata/rt-polarity.pos", "r", encoding='utf-8', errors='ignore').read().split("\n")[:-1]
      pos_list = []
      for p in pos:
        if len(p.split()) < 5:
          continue
        pos_list.append((p, 0))
      neg = codecs.open("rt-polaritydata/rt-polarity.neg", "r", encoding='utf-8', errors='ignore').read().split("\n")[:-1]
      neg_list = []
      for n in neg:
        if len(n.split()) < 5:
          continue
        pos_list.append((n, 1))
      p_train, p_val = self._split_train_val(pos_list)
      n_train, n_val = self._split_train_val(neg_list)
      train = p_train + n_train
      val = p_val + n_val
      pickle.dump(train, open("rt-polaritydata/train.pkl", "wb", True))
      pickle.dump(val, open("rt-polaritydata/val.pkl", "wb", True))
    
    if self.phase == 'train':
      return pickle.load(open("rt-polaritydata/train.pkl", "rb", True))
    elif self.phase == 'val':
      return pickle.load(open("rt-polaritydata/val.pkl", "rb", True))

  def _split_train_val(self, data_list, val_rate=0.1, random_seed=1, shuffle=True):
    data_len = len(data_list)
    val_len = int(data_len * val_rate)
    if random_seed:
      random.seed(random_seed)
    if shuffle:
      data_idx = random.sample(range(data_len), data_len)
    else:
      data_idx = list(range(data_len))
    val_data   = [data_list[idx] for idx in data_idx[:val_len]]
    train_data = [data_list[idx] for idx in data_idx[val_len:]]
    return train_data, val_data

  def _get_wv(self, word):
    wv = self.wordvec.wv
    if word in wv:
      return wv[word]
    dims = self.wordvec.get_dim()
    return np.random.rand(dims)

if __name__ == "__main__":
  train_data = PolarityDataset(phase='train')

  # test a batch
  batch_size = 4
  for i in range(batch_size):
    sample = train_data[i]
    print("sample %d," % i, sample['X'].size(), sample['Y'].size())
    assert sample['X'].size()[0] == train_data.wordvec.get_dim()
