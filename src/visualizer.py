# -*- coding: utf8

from __future__ import print_function

from scipy.misc import imsave
from PIL import Image
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

model_path = "model/RMSprop-google-batch25-epoch4000-lr0.0001-momentum0.9-wdecay0.0-kernels128"
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
print("Using cache")
visualizor = Visualizor(state_dict=cnn_model.state_dict(), args=args)
criterion = nn.BCEWithLogitsLoss()

def val(epoch):
  ''' chech if the pre-trained weight loaded
  '''
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

def visualize(x, layer, filter_idx=None, lr=1e-7, iters=50, verbose=False):
  visualizor.eval()
  assert layer + '.weight' in visualizor.state_dict().keys(), 'layer\'s weight not in model'

  old_score = -float('Inf')
  for iter in range(iters):
    visualizor.zero_grad()
    inputs = Parameter(torch.from_numpy(x).float(), requires_grad=True)
    if inputs.grad is not None:
      inputs.grad.data.zero_()
    output = visualizor(inputs)[layer]
    if filter_idx != None:
      score = output.data[..., filter_idx].sum()
    else:
      score = output.data.sum()
    if old_score >= score:
      break
    else:
      old_score = score
    loss = criterion(output, Variable(torch.zeros(output.size())))  # just random initial a loss object
    loss.data = torch.FloatTensor([score])
    loss.backward(retain_graph=True)
    grad = inputs.grad.data.cpu().numpy()
    #print(x.sum(), score, grad.shape, grad.sum())
    x[..., :] += lr * grad[..., :]  # gradient ascent
    #print(x.sum())
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

  def save_img(name, x):
    if x.shape[-1] == 1:
      x = x.reshape(x.shape[0], x.shape[1])
    x = Image.fromarray((x * 255).astype(np.uint8))
    x.save(name)

  def deprocess_image(x):
    scale = x.max() - x.min()
    x -= x.min()
    x /= scale
    x *= 255
    x = x.transpose((1, 2, 0))
    return x

  def deprocess_norm(x):
    x -= x.mean()
    x /= (x.std() + 1e-6)
    x = x.transpose((1, 2, 0))
    return x

  # util function to convert a tensor into a valid image
  def deprocess_image_keras(x):
    # normalize tensor: center on 0., ensure std is 0.1
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1

    # clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)

    # convert to RGB array
    x *= 255
    x = x.transpose((1, 2, 0))
    x = np.clip(x, 0, 255).astype('uint8')
    return x

  # test 5 samples
  # cls = ['pos', 'neg']
  # for i in range(5):
  #   sample = train_data[i]
  #   print("cls: {}, text: {}".format(cls[np.argmax(sample['Y'].numpy())], sample['text']))
  #   x = np.expand_dims(sample['X'].numpy(), axis=0)
  #   sample['Y'] = torch.from_numpy(np.expand_dims(sample['Y'].numpy(), axis=0))
  #   target = Variable(sample['Y'])
  #   _, x_grad = visualize(x, 'conv1', 0)
  #   format_visualization(x_grad, 'conv1_0')
  #   _, x_grad = visualize(x, 'conv2', 0)
  #   format_visualization(x_grad, 'conv2_0')
  #   _, x_grad = visualize(x, 'conv3', 0)
  #   format_visualization(x_grad, 'conv3_0')
  #   _, x_grad = visualize(x, 'fc', 0)
  #   format_visualization(x_grad, 'fc_0')
  #   _, x_grad = visualize(x, 'fc', 1)
  #   format_visualization(x_grad, 'fc_1')
  #   x_grad = saliency(x, target)
  #   format_visualization(x_grad, 'fc')

  # test img that activate convs
  cases = [('conv1', 0), ('conv1', 1), ('conv1', 50), ('conv1', None), ('conv2', 0), ('conv2', 1), ('conv2', 50), ('conv2', None), ('conv3', 0), ('conv3', 1), ('conv3', 50), ('fc', 0), ('fc', 1), ('fc', None)]
  for case in cases:
    layer = case[0]
    idx = case[1]

    if layer == 'fc':
      x = np.zeros((3, args.wv_dims, 100))
    else:
      x = np.zeros((3, args.wv_dims, 10))
    x_max, _ = visualize(x, layer, filter_idx=idx)
    x_max = deprocess_norm(x_max)
    if idx == None:
      save_img('{}.png'.format(layer), x_max)
    else:
      save_img('{}_{}.png'.format(layer, idx), x_max)

    x = np.zeros((3, args.wv_dims, 10))
    _, grad = visualize(x, layer, filter_idx=idx, iters=1)
    grad = deprocess_norm(grad)
    if idx == None:
      save_img('{}_grad.png'.format(layer), grad)
    else:
      save_img('{}_{}_grad.png'.format(layer, idx), grad)
