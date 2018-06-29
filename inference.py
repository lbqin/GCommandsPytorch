from __future__ import print_function
import logging
import time
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

logger = logging.getLogger(__name__)
logging.basicConfig(level = logging.DEBUG,
                    format = '%(asctime)s[%(levelname)s] ---- %(message)s',
                    )

def inference(loader, spects, model, classes, cuda=False, verbose=True):
    model.eval()
    test_loss = 0
    correct = 0
    f = open("result/result.txt", "w")

    sentences = []
    #spects = loader.spects
    for line in spects:
        line = line[0]
        sentences.append(line.strip().split("/")[-1].split()[0].replace("fb", "pcm"))
    sentences = np.array(sentences)

    classes = np.array(classes)
    start = 0
    size = 0
    epoch_tic = time.time()
    for data, target in loader:
        if cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)

        size += target.size(0)
        sent = sentences[start:size]
        start = size

        output = model(data)
        test_loss += F.nll_loss(output, target, size_average=False).data[0]  # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()
        index = pred.numpy().astype(np.int32)
        index = np.array(index.reshape(len(index)))
        labels = classes[index]
        for s, label in zip(sent, labels):
            f.write(s + "\t" + label + "\n")

    test_loss /= len(loader.dataset)
    epoch_toc = time.time()
    epoch_time = epoch_toc - epoch_tic

    f.close()
    if verbose:
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(loader.dataset), 100. * correct / len(loader.dataset)))
    return test_loss