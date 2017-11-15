# -*- coding: utf8

from __future__ import print_function
import argparse
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
import numpy as np
from polarity_loader import PolarityDataset
from model import CNNSentanceClassifier as CNNSC

# Training settings
parser = argparse.ArgumentParser(description='PyTorch CNN Sentence Classification')
parser.add_argument('--batch-size', type=int, default=50,
                    help='input batch size for training (default: 50)')
parser.add_argument('--test-batch-size', type=int, default=1,
                    help='input batch size for testing (default: 1)')
parser.add_argument('--n_class', type=int, default=2,
                    help='number of class (default: 2)')
parser.add_argument('--epochs', type=int, default=100,
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=1e-3,
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5,
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--w-decay', type=float, default=0.,
                    help='L2 norm (default: 0)')
parser.add_argument('--cuda', type=int, default=1,
                    help='using CUDA training')
parser.add_argument('--multi-gpu', action='store_true', default=False,
                    help='using multi-gpu')
parser.add_argument('--seed', type=int, default=1,
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=50,
                    help='how many batches to wait before logging training status')
args = parser.parse_args()
args.cuda = args.cuda and torch.cuda.is_available()
print('args:', args)

torch.manual_seed(args.seed)
if args.cuda:
  torch.cuda.manual_seed(args.seed)

train_data   = PolarityDataset(phase='train')
train_loader = DataLoader(train_data, batch_size=1, shuffle=True, num_workers=4)
val_data     = PolarityDataset(phase='val')
val_loader   = DataLoader(val_data, batch_size=1, shuffle=False, num_workers=4)

# load 
cnn_model = CNNSC(wv_dims=train_data.wordvec.get_dim(), n_class=args.n_class)
if args.cuda:
  ts = time.time()
  cnn_model = cnn_model.cuda()
  if args.multi_gpu:
    num_gpu = list(range(torch.cuda.device_count()))
    cnn_model = nn.DataParallel(cnn_model, device_ids=num_gpu)
  print("Finish cuda loading, time elapsed {}".format(time.time() - ts))

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.SGD(cnn_model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.w_decay)

accs = np.zeros(args.batch_size)


def train(epoch):
  cnn_model.train()
  train_loss = 0.
  iter = 0
  for idx, batch in enumerate(train_loader):
    optimizer.zero_grad()
    if args.cuda:
      batch['X'] = batch['X'].cuda()
      batch['Y'] = batch['Y'].cuda()
    inputs, target = Variable(batch['X']), Variable(batch['Y'])
    output = cnn_model(inputs)
    loss = criterion(output, target)

    if idx % args.batch_size == (args.batch_size - 1):
      iter += 1
      loss.data[0] += train_loss
      loss.data[0] /= args.batch_size
      loss.backward()
      optimizer.step()
      if iter % args.log_interval == 0:
        print("Training epoch {}, iter {}, loss {}".format(epoch, iter, loss.data[0]))
      train_loss = 0.
    else:
      train_loss += loss.data[0]


def val(epoch):
  cnn_model.eval()
  val_loss = 0.
  correct = 0
  for idx, batch in enumerate(val_loader):
    if args.cuda:
      batch['X'] = batch['X'].cuda()
      batch['Y'] = batch['Y'].cuda()
    inputs, target = Variable(batch['X']), Variable(batch['Y'])
    output = cnn_model(inputs)
    val_loss += criterion(output, target).data[0]
    pred = np.argmax(output.data.cpu().numpy(), axis=1)
    target = np.argmax(target.data.cpu().numpy(), axis=1)
    correct += (pred == target).sum()

  val_loss /= len(val_data)
  acc = correct / len(val_data)
  accs[epoch] = acc
  np.save("accuracy", accs)
  print("Validating epoch {}, val_loss {}, acc {:.4f}({}/{})".format(epoch, val_loss, acc, correct, len(val_data)))


if __name__ == "__main__":
  val(0)  # test initial performance
  print("Strat training")
  for epoch in range(args.epochs):
    ts = time.time()
    train(epoch)
    val(epoch)
    print("Finish epoch {}, time elapsed {}".format(epoch, time.time() - ts))
