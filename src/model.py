# -*- coding: utf-8 -*-

from __future__ import print_function

import torch
import torch.nn as nn
from torch.autograd import Variable


class CNNSentanceClassifier(nn.Module):
  
  def __init__(self, wv_dims, n_class, channel=100):
    super().__init__()
    self.wv_dims = wv_dims
    self.n_class = n_class
    self.channel = channel

    self.conv1 = nn.Conv1d(self.wv_dims, self.channel, 3)
    self.conv2 = nn.Conv1d(self.wv_dims, self.channel, 4)
    self.conv3 = nn.Conv1d(self.wv_dims, self.channel, 5)

    self.drop = nn.Dropout(p=0.5, inplace=True)
    self.fc = nn.Linear(self.channel*3, self.n_class)

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
    x1 = self.conv1(x)  # shape=(N, self.channel, L')
    x2 = self.conv2(x)  # shape=(N, self.channel, L'')
    x3 = self.conv3(x)  # shape=(N, self.channel, L''')

    m1 = nn.MaxPool1d(x1.size()[-1])  # kernel size = L'
    m2 = nn.MaxPool1d(x2.size()[-1])  # kernel size = L''
    m3 = nn.MaxPool1d(x3.size()[-1])  # kernel size = L'''

    x1 = m1(x1).view(-1, self.channel)  # shape=(N, self.channel)
    x2 = m2(x2).view(-1, self.channel)  # shape=(N, self.channel)
    x3 = m3(x3).view(-1, self.channel)  # shape=(N, self.channel)

    x_cat = torch.cat((x1, x2, x3), 1)  # shape=(N, self.channel*3)
    out   = self.fc(self.drop(x_cat))   # shape=(N, self.n_class)

    return out


if __name__ == "__main__":
  # check output shape
  batch, wv_dims, sent_len, n_class = 10, 100, 17, 2
  model = CNNSentanceClassifier(wv_dims=wv_dims, n_class=n_class)
  input = torch.autograd.Variable(torch.randn(batch, wv_dims, sent_len))
  output = model(input)
  assert len(output.size()) == 2
  assert output.size()[0] == batch
  assert output.size()[1] == n_class
