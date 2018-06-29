from __future__ import print_function
import torch.nn.functional as F
from torch.autograd import Variable


def train(loader, model, optimizer, epoch, cuda, log_interval, verbose=True):
    model.train()
    global_epoch_loss = 0
    for batch_idx, (data, target) in enumerate(loader):
        if cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        global_epoch_loss += loss.data[0]
        if verbose:
            if batch_idx % log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(loader.dataset), 100.
                    * batch_idx / len(loader), loss.data[0]))
    train_loss = global_epoch_loss / len(loader.dataset)
    if verbose:
        print('\ntrain set: Average loss: {:.4f}\n'.format(train_loss))
    return train_loss


def valid(loader, model, cuda, verbose=True):
    model.eval()
    valid_loss = 0
    correct = 0
    for data, target in loader:
        if cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        valid_loss += F.nll_loss(output, target, size_average=False).data[0]  # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

        valid_loss /= len(loader.dataset)
        accuracy = 100. * float(correct) / float(len(loader.dataset))
    if verbose:
        print('\nValid set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
            valid_loss, correct, len(loader.dataset), accuracy))
    return valid_loss, accuracy
