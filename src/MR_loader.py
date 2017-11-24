# -*- coding: utf-8 -*-

from __future__ import print_function

from six.moves import cPickle as pickle
from six.moves import urllib
import tarfile
import numpy as np
import codecs
import random
import math
import re
import os

import torch
from torch.utils.data import Dataset
from wordvec import WordVec


def clean_str(string):
  ''' Tokenization/string cleaning for all datasets except for SST.
      Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
  '''
  string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
  string = re.sub(r"\'s", " \'s", string)
  string = re.sub(r"\'ve", " \'ve", string)
  string = re.sub(r"n\'t", " n\'t", string)
  string = re.sub(r"\'re", " \'re", string)
  string = re.sub(r"\'d", " \'d", string)
  string = re.sub(r"\'ll", " \'ll", string)
  string = re.sub(r",", " , ", string)
  string = re.sub(r"!", " ! ", string)
  string = re.sub(r"\(", " \( ", string)
  string = re.sub(r"\)", " \) ", string)
  string = re.sub(r"\?", " \? ", string)
  string = re.sub(r"\s{2,}", " ", string)
  return string.strip()


class MovieReviewDataset(Dataset):

  data_dir = 'data'

  def __init__(self, phase, wv_type='glove', val_rate=0.1):
    '''
      Parameters
      ----------
      phase   : phase of the task, equals to one in {'train', 'val'}
      wv_type : word vector type, equals to one in {'google', 'glove', 'self'}
    '''
    self.val_rate = val_rate
    self.phase = phase
    self.data, self.word_to_idx = self.get_data(self.phase)
    self.wordvec = WordVec(wv_type)

  def __len__(self):
    return len(self.data)

  def __getitem__(self, idx):
    text = self.data[idx][0]
    sent = np.array([self.word_to_idx[word] for word in text.split()])  # get every word's idx for given sentence
    #sent = torch.from_numpy(sent).float()
    cls = self.data[idx][1]
    target = np.array([cls])
    #target = torch.from_numpy(target).float()
    sample = {'X': sent, 'Y': target, 'text': text}
    return sample

  def get_data(self, phase):
    if not os.path.exists(self.data_dir):
      self._download_data()
    if not os.path.exists(os.path.join(self.data_dir, "train.pkl")) or not os.path.exists(os.path.join(self.data_dir, "val.pkl")) or not os.path.exists(os.path.join(self.data_dir, "wtoi.pkl")):
      pos = codecs.open(os.path.join(self.data_dir, "rt-polaritydata/rt-polarity.pos"), "r", encoding='utf-8', errors='ignore').read().split("\n")[:-1]
      pos_list = []
      for p in pos:
        p = clean_str(p)
        if len(p.split()) < 5:
          continue
        pos_list.append((p, 0))
      neg = codecs.open(os.path.join(self.data_dir, "rt-polaritydata/rt-polarity.neg"), "r", encoding='utf-8', errors='ignore').read().split("\n")[:-1]
      neg_list = []
      for n in neg:
        n = clean_str(n)
        if len(n.split()) < 5:
          continue
        neg_list.append((n, 1))
      word_to_idx = {'@pad': 0}
      for sent in pos_list + neg_list:
        for word in sent[0].split():
          if word not in word_to_idx:
            word_to_idx[word] = len(word_to_idx)
      p_train, p_val = self._split_train_val(pos_list)
      n_train, n_val = self._split_train_val(neg_list)
      train = p_train + n_train
      val = p_val + n_val
      pickle.dump(train, open(os.path.join(self.data_dir, "train.pkl"), "wb", True))
      pickle.dump(val, open(os.path.join(self.data_dir, "val.pkl"), "wb", True))
      pickle.dump(word_to_idx, open(os.path.join(self.data_dir, "wtoi.pkl"), "wb", True))
    
    if self.phase == 'train':
      d = pickle.load(open(os.path.join(self.data_dir, "train.pkl"), "rb", True))
    elif self.phase == 'val':
      d = pickle.load(open(os.path.join(self.data_dir, "val.pkl"), "rb", True))
    wtoi = pickle.load(open(os.path.join(self.data_dir, "wtoi.pkl"), "rb", True))
    return (d, wtoi)

  def get_dict_wv(self):
    wv = self.wordvec.wv
    dims = self.wordvec.get_dim()
    dict_wv = np.zeros((len(self.word_to_idx), dims))  # |V| * word_vec_dim
    oov = 0
    for word in self.word_to_idx.keys():
      if word in wv:
        idx = self.word_to_idx[word]
        dict_wv[idx] = wv[word]
      else:
        oov += 1
    self.oov = oov
    return dict_wv

  def _split_train_val(self, data_list, random_seed=1, shuffle=True):
    data_len = len(data_list)
    val_len = int(data_len * self.val_rate)
    if random_seed:
      random.seed(random_seed)
    if shuffle:
      data_idx = random.sample(range(data_len), data_len)
    else:
      data_idx = list(range(data_len))
    val_data   = [data_list[idx] for idx in data_idx[:val_len]]
    train_data = [data_list[idx] for idx in data_idx[val_len:]]
    return train_data, val_data

  def _get_word_wv(self, word):
    wv = self.wordvec.wv
    if word in wv:
      return wv[word]
    dims = self.wordvec.get_dim()
    return np.random.rand(dims)

  def _download_data(self):
    print("Downloading data")
    url = "https://www.cs.cornell.edu/people/pabo/movie-review-data/rt-polaritydata.tar.gz"
    filename = "rt-polaritydata.tar"
    urllib.request.urlretrieve(url, filename)
    with tarfile.open(filename, 'r') as tfile:
      tfile.extractall(self.data_dir)
    os.remove(filename)


class MRLoader:

  def __init__(self, dataset):
    self.dataset = dataset
    self.data_len = len(self.dataset)
    self.index_list = self.shuffle_index()
    self.curr_index = 0

  def shuffle_index(self):
    return random.sample(range(self.data_len), self.data_len)

  def get_batch_num(self, batch_size):
    return math.floor(self.data_len / batch_size)

  def next_batch(self, batch_size):
    batch_index = self._gen_batch_index(batch_size)
    batch_X = [self.dataset[idx]['X'] for idx in batch_index]
    batch_Y = [self.dataset[idx]['Y'] for idx in batch_index]
    batch = self._pad_sequence(batch_X, batch_Y)
    return batch

  def _gen_batch_index(self, batch_size):
    if self.curr_index + batch_size > self.data_len:
      batch_index = self.index_list[self.curr_index:self.data_len]
      self.index_list = self.shuffle_index()
      remain_size = batch_size - (self.data_len - self.curr_index)
      batch_index += self.index_list[:remain_size]
      self.curr_index = remain_size
    else:
      batch_index = self.index_list[self.curr_index:self.curr_index+batch_size]
      self.curr_index += batch_size
    return batch_index

  def _pad_sequence(self, X, Y):
    max_len = max([len(x) for x in X])
    ret = {
      'X': np.zeros((len(X), max_len)),
      'Y': np.array(Y)
    }
    for idx, x in enumerate(X):
      if len(x) < max_len:
        pad = np.zeros(max_len-len(x))
        ret[idx] = np.append(x, pad)  # pad with zero
    for key in ret.keys():
      ret[key] = torch.from_numpy(ret[key]).long()
    return ret


if __name__ == "__main__":
  train_data = MovieReviewDataset(phase='train', wv_type='google')
  embed = train_data.get_dict_wv()
  print('oov {}, |V| {}'.format(train_data.oov, embed.shape[0]))

  # test a batch
  batch_size = 4
  for i in range(batch_size):
    sample = train_data[i]
    print("sample %d," % i, sample['X'].shape, sample['Y'].shape)

  # test dataloader
  train_loader = MRLoader(dataset=train_data)
  n_batch = train_loader.get_batch_num(batch_size=batch_size)
  for nb in range(n_batch):
    batch = train_loader.next_batch(batch_size=batch_size)
    print("batch {}, x.size {}, y.size {}".format(nb, batch['X'].size(), batch['Y'].size()))
    assert batch['X'].size()[0] == batch_size
    assert batch['Y'].size()[0] == batch_size
    if nb == 3:
      break
  print("Pass Test")
