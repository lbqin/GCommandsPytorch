from __future__ import print_function
import argparse
import torch
import torch.optim as optim
from gcommand_loader import GCommandLoader
import numpy as np
from model import LeNet, VGG
from train import train, valid
from inference import inference
import os
from gcommand_loader import get_classes


# Training settings
parser = argparse.ArgumentParser(
    description='ConvNets for Speech Commands Recognition')
parser.add_argument('--train_path', default='gcommands/train',
                    help='path to the train data folder')
parser.add_argument('--test_path', default='gcommands/test',
                    help='path to the test data folder')
parser.add_argument('--valid_path', default='gcommands/valid',
                    help='path to the valid data folder')
parser.add_argument('--batch_size', type=int, default=100,
                    metavar='N', help='training and valid batch size')
parser.add_argument('--test_batch_size', type=int, default=100,
                    metavar='N', help='batch size for testing')
parser.add_argument('--arc', default='LeNet',
                    help='network architecture: LeNet, VGG11, VGG13, VGG16, VGG19')
parser.add_argument('--epochs', type=int, default=100,
                    metavar='N', help='number of epochs to train')
parser.add_argument('--lr', type=float, default=0.001,
                    metavar='LR', help='learning rate')
parser.add_argument('--momentum', type=float, default=0.9,
                    metavar='M', help='SGD momentum, for SGD only')
parser.add_argument('--optimizer', default='adam',
                    help='optimization method: sgd | adam')
parser.add_argument('--cuda', default=True, help='enable CUDA')
parser.add_argument('--seed', type=int, default=1234,
                    metavar='S', help='random seed')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='num of batches to wait until logging train status')
parser.add_argument('--patience', type=int, default=5, metavar='N',
                    help='how many epochs of no loss improvement should we wait before stop training')

# feature extraction options
parser.add_argument('--window_size', default=.02,
                    help='window size for the stft')
parser.add_argument('--window_stride', default=.01,
                    help='window stride for the stft')
parser.add_argument('--window_type', default='hamming',
                    help='window type for the stft')
parser.add_argument('--normalize', default=True,
                    help='boolean, wheather or not to normalize the spect')

parser.add_argument('--no-train', dest='train', action='store_false')
parser.add_argument('--checkpoint', default='checkpoint', metavar='CHECKPOINT', help='checkpoints directory')

args = parser.parse_args()

classes, _, class_to_idx = get_classes()
num_classes = len(classes)
print(num_classes)
# build model
if args.arc == 'LeNet':
    model = LeNet(num_classes)
elif args.arc.startswith('VGG'):
    model = VGG(args.arc, num_classes)
else:
    model = LeNet(num_classes)

if args.train:
    args.cuda = args.cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    # loading data
    train_dataset = GCommandLoader(args.train_path, window_size=args.window_size, window_stride=args.window_stride,
                                   window_type=args.window_type, normalize=args.normalize)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=20, pin_memory=args.cuda, sampler=None)

    valid_dataset = GCommandLoader(args.valid_path, window_size=args.window_size, window_stride=args.window_stride,
                                   window_type=args.window_type, normalize=args.normalize)
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=args.batch_size, shuffle=None,
        num_workers=20, pin_memory=args.cuda, sampler=None)

    if args.cuda:
        print('Using CUDA with {0} GPUs'.format(torch.cuda.device_count()))
        model = torch.nn.DataParallel(model).cuda()

    # define optimizer
    if args.optimizer.lower() == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
    elif args.optimizer.lower() == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.lr,
                              momentum=args.momentum)
    else:
        optimizer = optim.SGD(model.parameters(), lr=args.lr,
                              momentum=args.momentum)

    best_valid_loss = np.inf
    best_train_loss = np.inf
    best_valid_acc = 0
    iteration = 0
    epoch = 1


    # trainint with early stopping
    while (epoch < args.epochs + 1) and (iteration < args.patience):
        train_loss = train(train_loader, model, optimizer, epoch, args.cuda, args.log_interval)
        valid_loss, valid_acc = valid(valid_loader, model, args.cuda)

        if valid_acc <= best_valid_acc:
            iteration += 1
            print('Loss was not improved, iteration {0}'.format(str(iteration)))
        else:
            print('Saving model...')
            iteration = 0
            best_valid_acc = valid_acc
            # state = {
            #     'valid_acc': valid_acc,
            #     'valid_loss': valid_loss,
            #     'epoch': epoch,
            # }
            state = {
                'net': model.module if args.cuda else model,
                'acc': valid_loss,
                'epoch': epoch,
            }
            if not os.path.isdir(args.checkpoint):
                os.mkdir(args.checkpoint)
            #torch.save(state, './{}/ckpt.t7'.format(args.checkpoint))
            #torch.save(state, './checkpoint/ckpt.t7')
            torch.save(model.state_dict(), './{}/ckpt.t7'.format(args.checkpoint))
        epoch += 1
    print('best valid acc:', best_valid_acc)
else:
    # test model
    test_dataset = GCommandLoader(args.test_path, window_size=args.window_size, window_stride=args.window_stride,
                                  window_type=args.window_type, normalize=args.normalize)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=None,
        num_workers=20, sampler=None)

    state_dict = torch.load('./{}/ckpt.t7'.format(args.checkpoint), map_location=lambda storage, loc: storage)

    # create new OrderedDict that does not contain `module.`
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] # remove `module.`
        new_state_dict[name] = v
    # load params
    model.load_state_dict(new_state_dict)
    print(model)

    inference(test_loader, test_dataset.spects, model, classes)
