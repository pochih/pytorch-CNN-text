# -*- coding: utf8

from __future__ import print_function

import numpy as np
import argparse
import time
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader

from twitter_loader import TwitterDataset
from embed_model import SC_Embedding, Twitter_Embedding

parser = argparse.ArgumentParser(description='PyTorch CNN Sentence Classification')
# training configs
parser.add_argument('--optimizer', type=str, default='Adam',
                    help='training optimizer (default: Adam)')
parser.add_argument('--batch-size', type=int, default=50,
                    help='input batch size for training (default: 50)')
parser.add_argument('--test-batch-size', type=int, default=100,
                    help='input batch size for testing (default: 100)')
parser.add_argument('--bs-increase-interval', type=int, default=50,
                    help='how many epochs to wait before increase batch_size (default: 50)')
parser.add_argument('--bs-increase-rate', type=float, default=1,
                    help='batch_size increase rate (default: 1)')
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
                    help='how many batches to wait before logging training status (default: 50)')
parser.add_argument('--pretrained', type=int, default=0,
                    help='use pre-trained model (default: 0)')
# model
parser.add_argument('--kernels', type=int, default=100, 
                    help='kernels for each conv layer (default: 100)')
parser.add_argument('--dropout', type=float, default=0.5, 
                    help='probability for dropout (default: 0.5)')
# data
parser.add_argument('--wv-type', type=str, default='glove', 
                    help='word vector for training (default: glove)')
# device
parser.add_argument('--cuda', type=int, default=1,
                    help='using CUDA training')
parser.add_argument('--multi-gpu', action='store_true', default=False,
                    help='using multi-gpu')
args = parser.parse_args()
args.cuda = args.cuda and torch.cuda.is_available()
params = "Twitter-Embedding-{}-{}-batch{}-epoch{}-lr{}-momentum{}-wdecay{}-kernels{}-drop{}".format(args.optimizer, args.wv_type, args.batch_size, args.epochs, args.lr, args.momentum, args.w_decay, args.kernels, args.dropout)
print('args: {}\nparams: {}'.format(args, params))

# define result file & model file
result_dir = 'result'
model_dir = 'model'
for dir in [result_dir, model_dir]:
  if not os.path.exists(dir):
    os.makedirs(dir)

# load data
train_data   = TwitterDataset(phase='train', wv_type=args.wv_type)
train_loader = DataLoader(train_data, batch_size=1, shuffle=True, num_workers=4)
val_data     = TwitterDataset(phase='val', wv_type=args.wv_type)
val_loader   = DataLoader(val_data, batch_size=1, shuffle=False, num_workers=4)
test_data    = TwitterDataset(phase='test', wv_type=args.wv_type)
test_loader  = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=4)
args.wv_dims = train_data.wordvec.get_dim()  # get word embedding size

# get model
if args.pretrained:
  cnn_model = torch.load(os.path.join(model_dir, params))
  accs = np.load(os.path.join(result_dir, params)+'.npy')
  print("Using cache")
else:
  cnn_model = Twitter_Embedding(args, embed_weight=train_data.get_dict_wv())
  accs = np.zeros(args.epochs)
if args.cuda:
  ts = time.time()
  cnn_model = cnn_model.cuda()
  if args.multi_gpu:
    num_gpu = list(range(torch.cuda.device_count()))
    cnn_model = nn.DataParallel(cnn_model, device_ids=num_gpu)
  print("Finish cuda loading, time elapsed {}".format(time.time() - ts))

# define loss & optimizer
criterion = nn.CrossEntropyLoss()
if args.optimizer == 'Adam':
  optimizer = optim.Adam(cnn_model.parameters(), lr=args.lr, weight_decay=args.w_decay)
elif args.optimizer == 'SGD':
  optimizer = optim.SGD(cnn_model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.w_decay)
elif args.optimizer == 'RMSprop':
  optimizer = torch.optim.RMSprop(cnn_model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.w_decay)


def train(epoch):
  cnn_model.train()
  train_loss = 0.
  iter = 0
  for idx, batch in enumerate(train_loader):
    optimizer.zero_grad()
    if args.cuda:
      batch['X'] = batch['X'].cuda()
      batch['Y'] = batch['Y'].cuda()
    batch['Y'] = batch['Y'].view(-1)
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
    batch['Y'] = batch['Y'].view(-1)
    inputs, target = Variable(batch['X']), Variable(batch['Y'])
    output = cnn_model(inputs)
    val_loss += criterion(output, target).data[0]
    pred = np.argmax(output.data.cpu().numpy(), axis=1)
    target = target.data.cpu().numpy()
    correct += (pred == target).sum()

  val_loss /= len(val_data)
  acc = correct / len(val_data)
  if acc >= np.max(accs):
    torch.save(cnn_model, os.path.join(model_dir, params))
    infer()
  accs[epoch] = acc
  np.save(os.path.join(result_dir, params), accs)
  print("Validating epoch {}, val_loss {}, acc {:.4f}({}/{}), best {}".format(epoch, val_loss, acc, correct, len(val_data), np.max(accs)))


def infer():
  cnn_model.eval()
  with open(params + '.csv', 'w') as out:
    out.write('id,label\n')
    for idx, batch in enumerate(test_loader):
      if args.cuda:
        batch['X'] = batch['X'].cuda()
      inputs = Variable(batch['X'])
      output = cnn_model(inputs)
      pred = int(np.argmax(output.data.cpu().numpy(), axis=1).flatten())
      ID = batch['ID']
      out.write('{},{}\n'.format(ID, pred))


if __name__ == "__main__":
  #val(0)  # test initial performance before training

  print("Strat training")
  for epoch in range(args.epochs):
    # increase batch size, its similar to decrease the lr (ref: https://arxiv.org/abs/1711.00489)
    if epoch % args.bs_increase_interval == (args.bs_increase_interval - 1):
      args.batch_size = int(args.batch_size * args.bs_increase_rate)

    ts = time.time()
    train(epoch)
    val(epoch)
    print("Finish epoch {}, time elapsed {}".format(epoch, time.time() - ts))

  print("Best val acc {}".format(np.max(accs)))
