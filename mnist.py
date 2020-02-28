'''Hello Pytorch MNIST sample!'''

__author__ = "TAKAHASHI_Masaharu"
__version__ = "1.0.0"
__date__ = "2020/2/27"


import argparse
import logging
import numpy as np
from logging import Formatter, StreamHandler, getLogger
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import time

import MyModel
import strcolors


def argparser():
    parser = argparse.ArgumentParser(description='PyTorch Example')
    parser.add_argument('-e', '--epoch', default=10,
                        help='Epoch')
    parser.add_argument('--disable-cuda', action='store_true',
                        help='Disable CUDA')
    parser.add_argument('--debug', action='store_true',
                        help='Disable debug mode')
    args = parser.parse_args()
    args.device = torch.device('cpu')
    if not args.disable_cuda and torch.cuda.is_available():
        args.device = torch.device('cuda')

    return args

def logger_setting(args):
    logger = getLogger("Log")
    logger.setLevel(logging.DEBUG)
    stream_handler = StreamHandler()
    stream_handler.setLevel(logging.INFO)
    if args.debug:
        stream_handler.setLevel(logging.DEBUG)
    
    handler_format = Formatter('%(asctime)s - %(levelname)s\t%(message)s', datefmt='%Y/%m/%d %H:%M:%S')
    stream_handler.setFormatter(handler_format)
    logger.addHandler(stream_handler)
    return logger

def color_setting():
    return strcolors.pycolor()

def data_loader():
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, ), (0.5, ))])
    trainset = torchvision.datasets.MNIST(root='./data', 
                                            train=True,
                                            download=True,
                                            transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset,
                                                batch_size=100,
                                                shuffle=True,
                                                num_workers=2)
    
    testset = torchvision.datasets.MNIST(root='./data', 
                                            train=False, 
                                            download=True, 
                                            transform=transform)
    testloader = torch.utils.data.DataLoader(testset, 
                                                batch_size=100,
                                                shuffle=False, 
                                                num_workers=2)
    return trainloader, testloader
    x = np.random.random((10,2)) * 4 -2
    return x

def calculator(args, model, criterion, loader, is_train=False, optimizer=None):
    loss = 0.0001
    total = 0
    correct = 0
    for iterator, (inputs, labels) in enumerate(loader, 1):
        inputs = inputs.to(device=args.device)
        labels = labels.to(device=args.device)
        y = model(inputs)

        loss = criterion(y, labels)
        if is_train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        loss += loss.item()

        predicted = y.argmax(axis=1)
        correct += (predicted==labels).sum().item()
        total += labels.size(0)
        break #! debug mode
    return loss/iterator, correct/total

def main():
    start = time.time()
    args = argparser()
    color = color_setting()
    logger = logger_setting(args)
    logger.info(__doc__)
    #####
#    x = data_loder()
#    t = x[:,0] * x[:,0] + x[:,1] * x[:,1] > 1
#    x = torch.as_tensor(x, dtype=torch.float32)
#    t = torch.as_tensor(t, dtype=torch.int64)

    trainloader, testloader = data_loader()
    ####

    model = MyModel.CNN()
    model = model.to(device=args.device)
    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss().to(device=args.device)

    for epoch in range(1, args.epoch+1):
        train_loss,train_accuracy = calculator(args, model, criterion, trainloader, is_train=True, optimizer=optimizer)
        with torch.no_grad():
            test_loss, test_accuracy = calculator(args, model, criterion, testloader, is_train=False)

        epoch_message = f"epoch: {epoch}"
        train_loss_message = f"TrainLoss: {color.RED}{train_loss:.3f}{color.END}"
        train_accuracy_message = f"TrainAccuracy: {color.RED}{train_accuracy:.3f}{color.END}"

        test_loss_message = f"TestLoss: {color.RED}{test_loss:.3f}{color.END}"
        test_accuracy_message = f"TestAccuracy: {color.RED}{test_accuracy:.3f}{color.END}"

        logger.info(f"{epoch_message}\t{train_loss_message}    {train_accuracy_message}    {test_loss_message}    {test_accuracy_message}")
    elapsed_time = time.time() - start
    logger.debug(f"elapsed_time: {elapsed_time:.2f} [sec]")

if __name__ == '__main__':
    main()
