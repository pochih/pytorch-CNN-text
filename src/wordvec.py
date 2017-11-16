# -*- coding: utf8

from __future__ import print_function

from gensim.models import word2vec, KeyedVectors
from six.moves import cPickle as pickle
from scipy import spatial
import numpy as np
import codecs
import os


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
          f.write("\n".join(word_corpus))
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
      if not os.path.exists('wordvec/glove-6B.100d.pkl'):
        glove = open("wordvec/glove.6B/glove.6B.100d.txt", "r").read().split("\n")[:-1]
        word_vector = {line.split()[0]: np.array(line.split()[1:]).astype(np.float32) for line in glove}
        pickle.dump(word_vector, open('wordvec/glove-6B.100d.pkl', "wb", True))
      word_vector = pickle.load(open('wordvec/glove-6B.100d.pkl', "rb", True))
      dims = 100
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
  WV = WordVec(wv_type='glove')
  word_vector, _ = WV.get_wv()
  test_wv(word_vector, WV.wv_type)
  WV = WordVec(wv_type='google')
  word_vector, _ = WV.get_wv()
  test_wv(word_vector, WV.wv_type)
  WV = WordVec(wv_type='self')
  word_vector, _ = WV.get_wv()
  test_wv(word_vector, WV.wv_type)
  print("Pass Test")
