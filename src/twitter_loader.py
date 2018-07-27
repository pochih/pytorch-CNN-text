# -*- coding: utf-8 -*-

from __future__ import print_function

from six.moves import cPickle as pickle
import numpy as np
import codecs
import random
import math
import re
import os

import torch
from torch.utils.data import Dataset
from wordvec import TwitterWordVec


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


class TwitterDataset(Dataset):

  data_dir = 'twitter'

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
    self.wordvec = TwitterWordVec(wv_type)

  def __len__(self):
    return len(self.data)

  def __getitem__(self, idx):
    text = self.data[idx][0]
    sent = np.array([self.word_to_idx[word] if word in self.word_to_idx else 0 for word in text.split()])  # get every word's idx for given sentence
    if self.phase == 'test' and len(text.split()) < 5:
      sent = np.append(sent, np.zeros(5-len(sent)))
    #sent = torch.from_numpy(sent).long()
    cls = self.data[idx][1]
    target = np.array([cls])
    #target = torch.from_numpy(target).long()
    sample = {'X': sent, 'Y': target, 'text': text}
    if self.phase == 'test':
      sample['ID'] = self.data[idx][2]
    return sample

  def get_data(self, phase):
    if not os.path.exists(os.path.join(self.data_dir, "train.pkl")) or not os.path.exists(os.path.join(self.data_dir, "val.pkl")) or not os.path.exists(os.path.join(self.data_dir, "test.pkl")) or not os.path.exists(os.path.join(self.data_dir, "wtoi.pkl")):
      lines = codecs.open(os.path.join(self.data_dir, "training_label.txt"), "r", encoding='utf-8', errors='ignore').read().split("\n")[:-1]
      pos_list = []
      neg_list = []
      # loading training data
      for line in lines:
        line = line.split(" +++$+++ ")
        label = int(line[0])
        sent = line[1]
        sent = clean_str(sent)
        if len(sent.split()) < 5:
          continue
        if label == 0:
          neg_list.append((sent, 0))
        elif label == 1:
          pos_list.append((sent, 1))
      # build the dictionary
      word_to_idx = {'@pad': 0}
      for sent in pos_list + neg_list:
        for word in sent[0].split():
          if word not in word_to_idx:
            word_to_idx[word] = len(word_to_idx)
      # split the train/val
      p_train, p_val = self._split_train_val(pos_list)
      n_train, n_val = self._split_train_val(neg_list)
      train = p_train + n_train
      val = p_val + n_val
      # loading testing data
      lines = codecs.open(os.path.join(self.data_dir, "testing_data.txt"), "r", encoding='utf-8', errors='ignore').read().split("\n")[1:-1]
      test = []
      for line in lines:
        line = line.split(",")
        ID = int(line[0])
        sent = ",".join(line[1:])
        sent = clean_str(sent)
        test.append((sent, -1, ID))
      pickle.dump(train, open(os.path.join(self.data_dir, "train.pkl"), "wb", True))
      pickle.dump(val, open(os.path.join(self.data_dir, "val.pkl"), "wb", True))
      pickle.dump(test, open(os.path.join(self.data_dir, "test.pkl"), "wb", True))
      pickle.dump(word_to_idx, open(os.path.join(self.data_dir, "wtoi.pkl"), "wb", True))
    
    if self.phase == 'train':
      d = pickle.load(open(os.path.join(self.data_dir, "train.pkl"), "rb", True))
    elif self.phase == 'val':
      d = pickle.load(open(os.path.join(self.data_dir, "val.pkl"), "rb", True))
    elif self.phase == 'test':
      d = pickle.load(open(os.path.join(self.data_dir, "test.pkl"), "rb", True))
    wtoi = pickle.load(open(os.path.join(self.data_dir, "wtoi.pkl"), "rb", True))
    return (d, wtoi)

  def get_dict_wv(self):
    wv = self.wordvec.wv
    dims = self.wordvec.get_dim()
    dict_wv = np.random.uniform(-1, 1, size=(len(self.word_to_idx), dims))  # |V| * word_vec_dim
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


class TwitterLoader:

  def __init__(self, dataset, phase):
    self.dataset = dataset
    self.phase = phase
    self.data_len = len(self.dataset)
    self.index_list = self.shuffle_index()
    self.curr_index = 0

  def shuffle_index(self):
    if self.phase == 'train':
      return random.sample(range(self.data_len), self.data_len)
    else:
      return list(range(self.data_len))

  def get_batch_num(self, batch_size):
    return int(math.floor(self.data_len / batch_size))

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
      pad = np.zeros(max_len-len(x))
      ret['X'][idx] = np.append(x, pad)  # pad with zero
    for key in ret.keys():
      ret[key] = torch.from_numpy(ret[key]).long()
    return ret


if __name__ == "__main__":
  train_data = TwitterDataset(phase='train', wv_type='glove')
  embed = train_data.get_dict_wv()
  print('oov {}, |V| {}'.format(train_data.oov, embed.shape[0]))

  # test a batch
  batch_size = 4
  for i in range(batch_size):
    sample = train_data[i]
    print("sample %d," % i, sample['X'].shape, sample['Y'].shape)

  # test dataloader
  train_loader = TwitterLoader(dataset=train_data, phase='train')
  n_batch = train_loader.get_batch_num(batch_size=batch_size)
  for nb in range(n_batch):
    batch = train_loader.next_batch(batch_size=batch_size)
    print("batch {}, x.size {}, y.size {}".format(nb, batch['X'].size(), batch['Y'].size()))
    assert batch['X'].size()[0] == batch_size
    assert batch['Y'].size()[0] == batch_size
    if nb == 3:
      break
  print("Pass Test")
