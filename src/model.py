# -*- coding: utf-8 -*-

from __future__ import print_function

import torch
import torch.nn as nn
from torch.autograd import Variable


class CNNSentanceClassifier(nn.Module):
  
  def __init__(self, args):
    super().__init__()

    # random seed
    torch.manual_seed(args.seed)
    if args.cuda:
      torch.cuda.manual_seed(args.seed)

    self.wv_dims = args.wv_dims
    self.kernels = args.kernels
    self.n_class = args.n_class

    self.conv1 = nn.Conv1d(self.wv_dims, self.kernels, 3)
    self.conv2 = nn.Conv1d(self.wv_dims, self.kernels, 4)
    self.conv3 = nn.Conv1d(self.wv_dims, self.kernels, 5)

    self.selu = nn.SELU(True)
    self.drop = nn.Dropout(p=args.dropout, inplace=True)
    self.fc = nn.Linear(self.kernels*3, self.n_class)

  def forward(self, x):
    '''
      Parameters
      ----------
      x : a torch variable, which represent a sentence,
          x.data has shape (N, C, L)
           N equals to batch size
           C equals to word vector dims
           L equals to sentence length

      Returns
      -------
      output : a torch variable, which represent model's prediction
               output.data has shape (N, n_cls)
                N equals to batch size
                n_cls is the class num we want to discriminate
    '''
    x1 = self.selu(self.conv1(x))  # shape=(N, self.kernels, L')
    x2 = self.selu(self.conv2(x))  # shape=(N, self.kernels, L'')
    x3 = self.selu(self.conv3(x))  # shape=(N, self.kernels, L''')

    m1 = nn.MaxPool1d(x1.size()[-1])  # kernel size = L'
    m2 = nn.MaxPool1d(x2.size()[-1])  # kernel size = L''
    m3 = nn.MaxPool1d(x3.size()[-1])  # kernel size = L'''

    x1 = m1(x1).view(x.size()[0], self.kernels)  # shape=(N, self.kernels)
    x2 = m2(x2).view(x.size()[0], self.kernels)  # shape=(N, self.kernels)
    x3 = m3(x3).view(x.size()[0], self.kernels)  # shape=(N, self.kernels)

    x_cat = torch.cat((x1, x2, x3), 1)  # shape=(N, self.kernels*3)
    out   = self.fc(self.drop(x_cat))  # shape=(N, self.n_class)

    return out


class Visualizor(CNNSentanceClassifier):

  def __init__(self, state_dict, args):
    super().__init__(args)
    self.load_state_dict(state_dict)

  def forward(self, x):
    x1 = self.selu(self.conv1(x))  # shape=(N, self.kernels, L')
    x2 = self.selu(self.conv2(x))  # shape=(N, self.kernels, L'')
    x3 = self.selu(self.conv3(x))  # shape=(N, self.kernels, L''')

    m1 = nn.MaxPool1d(x1.size()[-1])  # kernel size = L'
    m2 = nn.MaxPool1d(x2.size()[-1])  # kernel size = L''
    m3 = nn.MaxPool1d(x3.size()[-1])  # kernel size = L'''

    x1 = m1(x1).view(x.size()[0], self.kernels)  # shape=(N, self.kernels)
    x2 = m2(x2).view(x.size()[0], self.kernels)  # shape=(N, self.kernels)
    x3 = m3(x3).view(x.size()[0], self.kernels)  # shape=(N, self.kernels)

    x_cat = torch.cat((x1, x2, x3), 1)  # shape=(N, self.kernels*3)
    fc   = self.fc(x_cat)  # shape=(N, self.n_class)

    return {
      'conv1': x1,
      'conv2': x2,
      'conv3': x3,
      'fc'   : fc
    }


if __name__ == "__main__":
  # check output shape
  batch, wv_dims, sent_len, n_class = 10, 100, 17, 5
  class Arg:
    seed    = 1
    cuda    = 0
    wv_dims = wv_dims
    kernels = 666
    n_class = n_class
    dropout = 0.5
  model = CNNSentanceClassifier(Arg())
  input = torch.autograd.Variable(torch.randn(batch, wv_dims, sent_len))
  output = model(input)
  assert len(output.size()) == 2
  assert output.size()[0] == batch
  assert output.size()[1] == n_class
  print("Pass Test")
