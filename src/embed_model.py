# -*- coding: utf-8 -*-

from __future__ import print_function

import torch
import torch.nn as nn
from torch.autograd import Variable


class SC_Embedding(nn.Module):
  
  def __init__(self, args, embed_weight):
    super().__init__()

    # random seed
    torch.manual_seed(args.seed)
    if args.cuda:
      torch.cuda.manual_seed(args.seed)

    self.embedding = nn.Embedding(embed_weight.shape[0], embed_weight.shape[1], padding_idx=0)
    self.embedding.weight.data.copy_(torch.from_numpy(embed_weight))

    self.wv_dims = args.wv_dims
    self.kernels = args.kernels
    self.n_class = args.n_class

    self.conv1 = nn.Conv1d(self.wv_dims, self.kernels, 3)
    self.conv2 = nn.Conv1d(self.wv_dims, self.kernels, 4)
    self.conv3 = nn.Conv1d(self.wv_dims, self.kernels, 5)

    self.selu = nn.SELU(True)

    self.fc1  = nn.Linear(self.kernels*3, 128)
    self.drop = nn.Dropout(p=args.dropout)
    self.fc2  = nn.Linear(128, self.n_class)

  def forward(self, x):
    '''
      Parameters
      ----------
      x : a torch variable, which represent a sentence,
          x.data has shape (N, L)
           N equals to batch size
           L equals to sentence length

      Returns
      -------
      output : a torch variable, which represent model's prediction
               output.data has shape (N, n_cls)
                N equals to batch size
                n_cls is the class num we want to discriminate
    '''
    embeds = self.embedding(x).permute(0, 2, 1) # shape=(N, W, L), C is the word vector size

    x1 = self.selu(self.conv1(embeds))  # shape=(N, self.kernels, L')
    x2 = self.selu(self.conv2(embeds))  # shape=(N, self.kernels, L'')
    x3 = self.selu(self.conv3(embeds))  # shape=(N, self.kernels, L''')

    m1 = nn.MaxPool1d(x1.size()[-1])  # kernel size = L'
    m2 = nn.MaxPool1d(x2.size()[-1])  # kernel size = L''
    m3 = nn.MaxPool1d(x3.size()[-1])  # kernel size = L'''

    x1 = m1(x1).view(x.size()[0], self.kernels)  # shape=(N, self.kernels)
    x2 = m2(x2).view(x.size()[0], self.kernels)  # shape=(N, self.kernels)
    x3 = m3(x3).view(x.size()[0], self.kernels)  # shape=(N, self.kernels)

    x_cat = torch.cat((x1, x2, x3), 1)  # shape=(N, self.kernels*3)
    fc1   = self.selu(self.fc1(x_cat))
    drop  = self.drop(fc1)
    out   = self.fc2(drop)  # shape=(N, self.n_class)

    return out

class Twitter_Embedding(nn.Module):
  
  def __init__(self, args, embed_weight):
    super().__init__()

    # random seed
    torch.manual_seed(args.seed)
    if args.cuda:
      torch.cuda.manual_seed(args.seed)

    self.embedding = nn.Embedding(embed_weight.shape[0], embed_weight.shape[1], padding_idx=0)
    self.embedding.weight.data.copy_(torch.from_numpy(embed_weight))

    self.wv_dims = args.wv_dims
    self.kernels = args.kernels
    self.n_class = args.n_class

    self.conv1 = nn.Conv1d(self.wv_dims, self.kernels, 2)
    self.conv2 = nn.Conv1d(self.wv_dims, self.kernels, 3)
    self.conv3 = nn.Conv1d(self.wv_dims, self.kernels, 4)
    self.conv4 = nn.Conv1d(self.wv_dims, self.kernels, 5)

    self.LReLU = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    self.fc1  = nn.Linear(self.kernels*4, 128)
    self.drop = nn.Dropout(p=args.dropout)
    self.fc2  = nn.Linear(128, self.n_class)

  def forward(self, x):
    '''
      Parameters
      ----------
      x : a torch variable, which represent a sentence,
          x.data has shape (N, L)
           N equals to batch size
           L equals to sentence length

      Returns
      -------
      output : a torch variable, which represent model's prediction
               output.data has shape (N, n_cls)
                N equals to batch size
                n_cls is the class num we want to discriminate
    '''
    embeds = self.embedding(x).permute(0, 2, 1) # shape=(N, W, L), C is the word vector size

    x1 = self.LReLU(self.conv1(embeds))  # shape=(N, self.kernels, L')
    x2 = self.LReLU(self.conv2(embeds))  # shape=(N, self.kernels, L'')
    x3 = self.LReLU(self.conv3(embeds))  # shape=(N, self.kernels, L''')
    x4 = self.LReLU(self.conv4(embeds))  # shape=(N, self.kernels, L'''')

    m1 = nn.MaxPool1d(x1.size()[-1])  # kernel size = L'
    m2 = nn.MaxPool1d(x2.size()[-1])  # kernel size = L''
    m3 = nn.MaxPool1d(x3.size()[-1])  # kernel size = L'''
    m4 = nn.MaxPool1d(x4.size()[-1])  # kernel size = L''''

    x1 = m1(x1).view(x.size()[0], self.kernels)  # shape=(N, self.kernels)
    x2 = m2(x2).view(x.size()[0], self.kernels)  # shape=(N, self.kernels)
    x3 = m3(x3).view(x.size()[0], self.kernels)  # shape=(N, self.kernels)
    x4 = m4(x4).view(x.size()[0], self.kernels)  # shape=(N, self.kernels)

    x_cat = torch.cat((x1, x2, x3, x4), 1)  # shape=(N, self.kernels*4)
    fc1   = self.LReLU(self.fc1(x_cat))
    drop  = self.drop(fc1)
    out   = self.fc2(drop)  # shape=(N, self.n_class)

    return out
