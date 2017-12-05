# -*- coding: utf8

from __future__ import print_function

from gensim.models import word2vec, KeyedVectors
from six.moves import cPickle as pickle
from scipy import spatial
import numpy as np
import codecs
import re
import os


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


class WordVec(object):
  ''' generate word vectors
  '''

  data_dir = 'data'

  def __init__(self, wv_type='glove'):
    '''
      argument
       wv_type: one of {'self', 'google', 'glove'}
         self:   use movie review dataset to train word embedding
         google: google pretrained 300 dims wordvec
         glove:  glove 100 dims wordvec trained by Wikipedia
    '''
    self.wv_type = wv_type
    self.wv, self.dims = self.get_wv()

  def get_wv(self):
    if self.wv_type == 'self':
      if not os.path.exists('wordvec/word_vector.bin'):
        pos = codecs.open(os.path.join(data_dir, "rt-polaritydata/rt-polarity.pos"), "r", encoding='utf-8', errors='ignore').read()
        neg = codecs.open(os.path.join(data_dir, "rt-polaritydata/rt-polarity.neg"), "r", encoding='utf-8', errors='ignore').read()
        pos_list = pos.split("\n")[:-1]
        neg_list = neg.split("\n")[:-1]

        # train word vector by dataset
        word_corpus = pos_list + neg_list
        tmp_file = 'all_words.txt'
        with codecs.open(tmp_file, 'w', encoding='utf-8', errors='ignore') as f:
          for sent in word_corpus:
            f.write("{}\n".format(clean_str(sent)))
        corpus = word2vec.Text8Corpus(tmp_file)
        word_vector = word2vec.Word2Vec(corpus, size=100)
        word_vector.wv.save_word2vec_format(u"wordvec/word_vector.bin", binary=True)
        os.remove(tmp_file)
      word_vector = KeyedVectors.load_word2vec_format('wordvec/word_vector.bin', binary=True).wv
      dims = 100
    elif self.wv_type == 'google':
      # load Google's pre-trained Word2Vec model.
      word_vector = KeyedVectors.load_word2vec_format('wordvec/GoogleNews-vectors-negative300.bin', binary=True).wv
      dims = 300
    elif self.wv_type == 'glove':
      if not os.path.exists('wordvec/glove-6B.300d.pkl'):
        glove = open("wordvec/glove.6B/glove.6B.300d.txt", "r").read().split("\n")[:-1]
        word_vector = {line.split()[0]: np.array(line.split()[1:]).astype(np.float32) for line in glove}
        pickle.dump(word_vector, open('wordvec/glove-6B.300d.pkl', "wb", True))
      word_vector = pickle.load(open('wordvec/glove-6B.300d.pkl', "rb", True))
      dims = 300
    return word_vector, dims

  def get_dim(self):
    return self.dims


class TwitterWordVec(object):
  ''' generate word vectors
  '''

  data_dir = 'twitter'

  def __init__(self, wv_type='glove'):
    '''
      argument
       wv_type: one of {'self', 'google', 'glove'}
         self:   use movie review dataset to train word embedding
         google: google pretrained 300 dims wordvec
         glove:  glove 100 dims wordvec trained by Wikipedia
    '''
    self.wv_type = wv_type
    self.wv, self.dims = self.get_wv()

  def get_wv(self):
    if self.wv_type == 'self':
      if not os.path.exists('wordvec/word_vector.bin'):
        word_corpus = []
        # add training data
        train = codecs.open(os.path.join(self.data_dir, "training_label.txt"), "r", encoding='utf-8', errors='ignore').read().split("\n")[:-1]
        for line in train:
          line = line.split(" +++$+++ ")
          label = int(line[0])
          sent = line[1]
          word_corpus.append(sent)
        # add unlabeled data
        unlabel = codecs.open(os.path.join(self.data_dir, "training_nolabel.txt"), "r", encoding='utf-8', errors='ignore').read().split("\n")[:-1]
        for line in unlabel:
          word_corpus.append(line)
        # train word vector by dataset
        tmp_file = 'all_words.txt'
        with codecs.open(tmp_file, 'w', encoding='utf-8', errors='ignore') as f:
          for sent in word_corpus:
            f.write("{}\n".format(clean_str(sent)))
        corpus = word2vec.Text8Corpus(tmp_file)
        word_vector = word2vec.Word2Vec(corpus, size=200)
        word_vector.wv.save_word2vec_format(u"wordvec/word_vector.bin", binary=True)
        os.remove(tmp_file)
      word_vector = KeyedVectors.load_word2vec_format('wordvec/word_vector.bin', binary=True).wv
      dims = 200
    elif self.wv_type == 'google':
      # load Google's pre-trained Word2Vec model.
      word_vector = KeyedVectors.load_word2vec_format('wordvec/GoogleNews-vectors-negative300.bin', binary=True).wv
      dims = 300
    elif self.wv_type == 'glove':
      if not os.path.exists('wordvec/glove-6B.300d.pkl'):
        glove = open("wordvec/glove.6B/glove.6B.300d.txt", "r").read().split("\n")[:-1]
        word_vector = {line.split()[0]: np.array(line.split()[1:]).astype(np.float32) for line in glove}
        pickle.dump(word_vector, open('wordvec/glove-6B.300d.pkl', "wb", True))
      word_vector = pickle.load(open('wordvec/glove-6B.300d.pkl', "rb", True))
      dims = 300
    return word_vector, dims

  def get_dim(self):
    return self.dims


def similarity(v1, v2):
  return 1 - spatial.distance.cosine(v1, v2)

def test_wv(wv, wv_type):
  print(wv_type)
  print("woman vs man", similarity(wv['woman'], wv['man']))
  print("woman vs queen", similarity(wv['woman'], wv['queen']))
  print("queen vs king", similarity(wv['king'],  wv['queen']))
  print("woman vs house", similarity(wv['woman'], wv['house']))
  print("woman vs beginning", similarity(wv['woman'], wv['beginning']))
  print("woman vs provocative", similarity(wv['woman'], wv['provocative']))

if __name__ == '__main__':
  # test three word embeddings
  WV = TwitterWordVec(wv_type='glove')
  word_vector, _ = WV.get_wv()
  test_wv(word_vector, WV.wv_type)
  WV = TwitterWordVec(wv_type='google')
  word_vector, _ = WV.get_wv()
  test_wv(word_vector, WV.wv_type)
  WV = TwitterWordVec(wv_type='self')
  word_vector, _ = WV.get_wv()
  test_wv(word_vector, WV.wv_type)
  print("Pass Test")
