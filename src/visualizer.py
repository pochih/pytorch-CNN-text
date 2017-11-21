# -*- coding: utf8

from __future__ import print_function

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
parser.add_argument('--wv-type', type=str, default='google', 
                    help='word vector for training (default: glove)')
# device
parser.add_argument('--cuda', type=int, default=0,
                    help='using CUDA training')
args = parser.parse_args()

model_path = "model/RMSprop-google-batch50-epoch3200-lr0.001-momentum0.9-wdecay0.0-kernels100"
print('model_path: {}'.format(model_path))

# load data
train_data   = PolarityDataset(phase='train', wv_type=args.wv_type)
train_loader = DataLoader(train_data, batch_size=1, shuffle=True, num_workers=4)
val_data     = PolarityDataset(phase='val', wv_type=args.wv_type)
val_loader   = DataLoader(val_data, batch_size=1, shuffle=False, num_workers=4)
args.wv_dims = train_data.wordvec.get_dim()  # get word embedding size

# get model
cnn_model = torch.load(model_path)
cnn_model = cnn_model.cpu()
print("Using cache")
visualizor = Visualizor(state_dict=cnn_model.state_dict(), args=args)
criterion = nn.BCEWithLogitsLoss()

def val(epoch):
  visualizor.eval()
  val_loss = 0.
  correct = 0
  for idx, batch in enumerate(val_loader):
    inputs, target = Variable(batch['X']), Variable(batch['Y'])
    output = visualizor(inputs)['fc']
    val_loss += criterion(output, target).data[0]
    pred = np.argmax(output.data.cpu().numpy(), axis=1)
    target = np.argmax(target.data.cpu().numpy(), axis=1)
    correct += (pred == target).sum()

  val_loss /= len(val_data)
  acc = correct / len(val_data)
  print("Validating epoch {}, val_loss {}, acc {:.4f}({}/{})".format(epoch, val_loss, acc, correct, len(val_data)))

def visualize(x, layer, filter_idx, lr=1e-2, iters=20, verbose=False):
  visualizor.eval()
  assert layer + '.weight' in visualizor.state_dict().keys(), 'layer\'s weight not in model'

  for iter in range(iters):
    visualizor.zero_grad()
    inputs = Parameter(torch.from_numpy(x).float(), requires_grad=True)
    if inputs.grad is not None:
      inputs.grad.data.zero_()
    output = visualizor(inputs)[layer]
    score = output.data[..., filter_idx]
    loss = criterion(output, Variable(torch.zeros(output.size())))  # just random initial a loss object
    loss.data = score
    loss.backward(retain_graph=True)
    grad = inputs.grad.data.cpu().numpy()
    x += lr * grad  # gradient ascent
    if iter % 10 == 0 and verbose:
      print('iter {}, score {}, grad.mean {}'.format(iter, score.sum(), grad.mean()))

  return x, grad

def saliency(x, target, verbose=False):
  cnn_model.eval()

  inputs = Parameter(torch.from_numpy(x).float(), requires_grad=True)
  if inputs.grad is not None:
    inputs.grad.data.zero_()
  output = cnn_model(inputs)
  loss = criterion(output, target)
  loss.backward(retain_graph=True)
  grad = inputs.grad.data.cpu().numpy()
  if verbose:
    print('grad.mean {}'.format(grad.mean()))

  return grad


if __name__ == '__main__':
  def format_visualization(x_grad, layer):
    x_grad = np.sum(x_grad.reshape(args.wv_dims, -1), axis=0)
    x_max = np.argmax(x_grad)
    x_min = np.argmin(x_grad)
    print('layer {}\nmax word: {}, min word: {}'.format(layer, sample['text'].split()[x_max], sample['text'].split()[x_min]))

  # test 5 samples
  cls = ['pos', 'neg']
  for i in range(5):
    sample = train_data[i]
    print("cls: {}, text: {}".format(cls[np.argmax(sample['Y'].numpy())], sample['text']))
    x = np.expand_dims(sample['X'].numpy(), axis=0)
    sample['Y'] = torch.from_numpy(np.expand_dims(sample['Y'].numpy(), axis=0))
    target = Variable(sample['Y'])
    _, x_grad = visualize(x, 'conv1', 0)
    format_visualization(x_grad, 'conv1_0')
    _, x_grad = visualize(x, 'conv2', 0)
    format_visualization(x_grad, 'conv2_0')
    _, x_grad = visualize(x, 'conv3', 0)
    format_visualization(x_grad, 'conv3_0')
    _, x_grad = visualize(x, 'fc', 0)
    format_visualization(x_grad, 'fc_0')
    _, x_grad = visualize(x, 'fc', 1)
    format_visualization(x_grad, 'fc_1')
    x_grad = saliency(x, target)
    format_visualization(x_grad, 'fc')
