# -*- coding: utf8

from __future__ import print_function

import argparse
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import os

from polarity_loader import PolarityDataset, PolarityLoader
from model import CNNSentanceClassifier as CNNSC

parser = argparse.ArgumentParser(description='PyTorch CNN Sentence Classification')
# training configs
parser.add_argument('--batch-size', type=int, default=50,
                    help='input batch size for training (default: 50)')
parser.add_argument('--test-batch-size', type=int, default=1,
                    help='input batch size for testing (default: 1)')
parser.add_argument('--bs-increase-interval', type=int, default=50,
                    help='how many epochs to wait before increase batch_size (default: 50)')
parser.add_argument('--bs-increase-rate', type=float, default=1.3,
                    help='batch_size increase rate (default: 1.3)')
parser.add_argument('--n-class', type=int, default=2,
                    help='number of class (default: 2)')
parser.add_argument('--epochs', type=int, default=100,
                    help='number of epochs to train (default: 100)')
parser.add_argument('--lr', type=float, default=1e-4,
                    help='learning rate (default: 0.0001)')
parser.add_argument('--momentum', type=float, default=0.9,
                    help='SGD momentum (default: 0.9)')
parser.add_argument('--w-decay', type=float, default=0.,
                    help='L2 norm (default: 0)')
parser.add_argument('--seed', type=int, default=1,
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=50,
                    help='how many batches to wait before logging training status')
# model
parser.add_argument('--kernels', type=int, default=100, 
                    help='kernels for each conv layer')
parser.add_argument('--dropout', type=float, default=0.5, 
                    help='probability for dropout (default: 0.5)')
# device
parser.add_argument('--cuda', type=int, default=1,
                    help='using CUDA training')
parser.add_argument('--multi-gpu', action='store_true', default=False,
                    help='using multi-gpu')
args = parser.parse_args()
args.cuda = args.cuda and torch.cuda.is_available()
params = "Adam-batch{}-epoch{}-lr{}-momentum{}-wdecay{}".format(args.batch_size, args.epochs, args.lr, args.momentum, args.w_decay)
print('args: {}\nparams: {}'.format(args, params))

# define result file & model file
result_dir = 'result'
model_dir = 'model'
for dir in [result_dir, model_dir]:
  if not os.path.exists(dir):
    os.makedirs(dir)
try:
  accs = np.load(os.path.join(result_dir, params)+'.npy')
except:
  accs = np.zeros(args.epochs)

# load data
train_data   = PolarityDataset(phase='train', wv_type='glove')
train_loader = PolarityLoader(dataset=train_data)
val_data     = PolarityDataset(phase='val', wv_type='glove')
val_loader   = PolarityLoader(dataset=val_data)
args.wv_dims = train_data.wordvec.get_dim()  # get word embedding size

# get model
try:
  cnn_model = torch.load(os.path.join(model_dir, params))
  print("Using cache")
except:
  cnn_model = CNNSC(args)
if args.cuda:
  ts = time.time()
  cnn_model = cnn_model.cuda()
  if args.multi_gpu:
    num_gpu = list(range(torch.cuda.device_count()))
    cnn_model = nn.DataParallel(cnn_model, device_ids=num_gpu)
  print("Finish cuda loading, time elapsed {}".format(time.time() - ts))

# define loss & optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(cnn_model.parameters(), lr=args.lr, weight_decay=args.w_decay)


def train(epoch):
  cnn_model.train()
  n_batch = train_loader.get_batch_num(batch_size=args.batch_size)
  for nb in range(n_batch):
    optimizer.zero_grad()
    batch = train_loader.next_batch(batch_size=args.batch_size)
    if args.cuda:
      batch['X'] = batch['X'].cuda()
      batch['Y'] = batch['Y'].cuda()
    inputs, target = Variable(batch['X']), Variable(batch['Y'])
    output = cnn_model(inputs)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()
    if nb % args.log_interval == 0:
      print("Training epoch {}, batch {}, loss {}".format(epoch, nb, loss.data[0]))
  torch.save(cnn_model, os.path.join(model_dir, params))


def val(epoch):
  cnn_model.eval()
  val_loss = 0.
  correct = 0
  n_batch = val_loader.get_batch_num(batch_size=args.test_batch_size)
  for _ in range(n_batch):
    optimizer.zero_grad()
    batch = val_loader.next_batch(batch_size=args.test_batch_size)
    if args.cuda:
      batch['X'] = batch['X'].cuda()
      batch['Y'] = batch['Y'].cuda()
    inputs, target = Variable(batch['X']), Variable(batch['Y'])
    output = cnn_model(inputs)
    loss = criterion(output, target)
    val_loss += loss.data[0]
    pred = np.argmax(output.data.cpu().numpy(), axis=1)
    target = np.argmax(target.data.cpu().numpy(), axis=1)
    correct += (pred == target).sum()

  val_loss /= len(val_data)
  acc = correct / len(val_data)
  accs[epoch] = acc
  np.save(os.path.join(result_dir, params), accs)
  print("Validating epoch {}, val_loss {}, acc {:.4f}({}/{})".format(epoch, val_loss, acc, correct, len(val_data)))


if __name__ == "__main__":
  val(0)  # test initial performance before training

  print("Strat training")
  for epoch in range(args.epochs):
    # increase batch size, its similar to decrease the lr (ref: https://arxiv.org/abs/1711.00489)
    if epoch % args.bs_increase_interval == args.bs_increase_interval - 1:
      args.batch_size = int(args.batch_size * args.bs_increase_rate)

    ts = time.time()
    train(epoch)
    val(epoch)
    print("Finish epoch {}, time elapsed {}".format(epoch, time.time() - ts))

  print("Best val acc {}".format(np.max(accs)))
