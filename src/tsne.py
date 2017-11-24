# -*- coding: utf8

from __future__ import print_function

from six.moves import cPickle as pickle
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import Parameter
from torch.autograd import Variable
from torch.utils.data import DataLoader

from polarity_loader import PolarityDataset, PolarityLoader
from model import CNNSentanceClassifier, Visualizor

parser = argparse.ArgumentParser(description='PyTorch CNN Sentence Classification')
# training configs
parser.add_argument('--n-class', type=int, default=2,
                    help='number of class (default: 2)')
parser.add_argument('--seed', type=int, default=1,
                    help='random seed (default: 1)')
# model
parser.add_argument('--kernels', type=int, default=100, 
                    help='kernels for each conv layer (default: 100)')
parser.add_argument('--dropout', type=float, default=0.5, 
                    help='probability for dropout (default: 0.5)')
# data
parser.add_argument('--wv-type', type=str, default='glove', 
                    help='word vector for training (default: glove)')
# device
parser.add_argument('--cuda', type=int, default=0,
                    help='using CUDA training')
args = parser.parse_args()

model_path = "model/Adam-batch25-epoch3000-lr0.0001-momentum0.9-wdecay0.0"
print('model_path: {}'.format(model_path))

# load data
train_data   = PolarityDataset(phase='train', wv_type=args.wv_type)
train_loader = DataLoader(train_data, batch_size=1, shuffle=False, num_workers=4)
val_data     = PolarityDataset(phase='val', wv_type=args.wv_type)
val_loader   = DataLoader(val_data, batch_size=1, shuffle=False, num_workers=4)
args.wv_dims = train_data.wordvec.get_dim()  # get word embedding size

# get model
cnn_model = torch.load(model_path)
cnn_model = cnn_model.cpu()
print(cnn_model)
print("Using cache")
visualizor = Visualizor(state_dict=cnn_model.state_dict(), args=args)
criterion = nn.BCEWithLogitsLoss()

def infer(input, layer):
  visualizor.eval()
  assert layer + '.weight' in visualizor.state_dict().keys(), 'layer\'s weight not in model'

  output = visualizor(inputs)[layer]
  return output.data.cpu().numpy()

def infer_filter(input, layer, filter_idx):
  visualizor.eval()
  assert layer + '.weight' in visualizor.state_dict().keys(), 'layer\'s weight not in model'

  output = visualizor(inputs)[layer]
  output = output.data[..., filter_idx]
  return np.array([output])

def plot_tSNE(input, name, s):
  pos_X = input[0][:, 0]
  pos_Y = input[0][:, 1]
  neg_X = input[1][:, 0]
  neg_Y = input[1][:, 1]

  plt.figure(figsize=(12, 9))
  plt.scatter(pos_X, pos_Y, c='b', label='positive', s=s)
  plt.scatter(neg_X, neg_Y, c='r', label='negative', s=s)

  plt.legend()
  plt.show()
  plt.savefig(name + '.png')


if __name__ == '__main__':
  train_result = {
    'conv1': {
      0: [],
      1: []
    },
    'conv2': {
      0: [],
      1: []
    },
    'conv3': {
      0: [],
      1: []
    },
    'fc': {
      0: [],
      1: []
    }
  }
  for idx, batch in enumerate(train_loader):
    label = int(np.argmax(batch['Y'][0].numpy()))  # 0=pos, 1=neg
    inputs = Variable(batch['X'])
    output = infer(inputs, 'conv1').flatten()
    train_result['conv1'][label].append(output)
    output = infer(inputs, 'conv2').flatten()
    train_result['conv2'][label].append(output)
    output = infer(inputs, 'conv3').flatten()
    train_result['conv3'][label].append(output)
    output = infer(inputs, 'fc').flatten()
    train_result['fc'][label].append(output)

  val_result = {
    'conv1': {
      0: [],
      1: []
    },
    'conv2': {
      0: [],
      1: []
    },
    'conv3': {
      0: [],
      1: []
    },
    'fc': {
      0: [],
      1: []
    }
  }
  for idx, batch in enumerate(val_loader):
    label = int(np.argmax(batch['Y'][0].numpy()))  # 0=pos, 1=neg
    inputs = Variable(batch['X'])
    output = infer(inputs, 'conv1').flatten()
    val_result['conv1'][label].append(output)
    output = infer(inputs, 'conv2').flatten()
    val_result['conv2'][label].append(output)
    output = infer(inputs, 'conv3').flatten()
    val_result['conv3'][label].append(output)
    output = infer(inputs, 'fc').flatten()
    val_result['fc'][label].append(output)

  for key in train_result:
    vectors = train_result[key][0] + train_result[key][1]
    vectors = TSNE(n_components=2).fit_transform(vectors)
    train_result[key][0] = vectors[:len(train_result[key][0])]
    train_result[key][1] = vectors[len(train_result[key][0]):]
  for key in val_result:
    vectors = val_result[key][0] + val_result[key][1]
    vectors = TSNE(n_components=2).fit_transform(vectors)
    val_result[key][0] = vectors[:len(val_result[key][0])]
    val_result[key][1] = vectors[len(val_result[key][0]):]

  pickle.dump(train_result, open('NN_traindata_result', 'wb', True))
  pickle.dump(val_result, open('NN_valdata_result', 'wb', True))

  train_result = pickle.load(open('NN_traindata_result', 'rb', True))
  val_result = pickle.load(open('NN_valdata_result', 'rb', True))

  for key in train_result:
    plot_tSNE(train_result[key], 'train-' + key, s=0.5)

  for key in val_result:
    plot_tSNE(val_result[key], 'val-' + key, s=5)
