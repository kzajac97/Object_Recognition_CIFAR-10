import sys
sys.path.insert(0,'..')

import mxnet as mx
from mxnet import nd
from mxnet import autograd
from mxnet import gluon
from mxnet import init

from mxnet.gluon import data as gluon_data
from mxnet.gluon import loss as gluon_loss
from mxnet.gluon import nn 
import mxnet.gluon.data.vision.datasets
from mxnet.gluon.data.vision.datasets import CIFAR10

import os
import pandas as pd
import shutil
import time

transform_train = gluon_data.vision.transforms.Compose([
    # Magnify the image to a square of 40 pixels in both height and width
    # gluon_data.vision.transforms.Resize(40),
    # Randomly crop a square image of 40 pixels in both height and width to
    # produce a small square of 0.64 to 1 times the area of the original
    # image, and then shrink it to a square of 32 pixels in both height and
    # width
    # gluon_data.vision.transforms.RandomResizedCrop(32, scale=(0.64, 1.0), ratio=(1.0, 1.0)),
    gluon_data.vision.transforms.RandomFlipLeftRight(),
    gluon_data.vision.transforms.ToTensor(),
    # Normalize each channel of the image
    gluon_data.vision.transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])

transform_test = gluon_data.vision.transforms.Compose([
    gluon_data.vision.transforms.ToTensor(),
    gluon_data.vision.transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])                                  

def Load_CIFAR10():
    data_train = gluon_data.vision.CIFAR10(train=True)
    data_test = gluon_data.vision.CIFAR10(train=False)

    return data_train, data_test

def Get_Iters(train,test,batch_size,workers):
    train_iter = gluon_data.DataLoader(train.transform_first(transform_train), batch_size, shuffle = True, num_workers = workers)
    test_iter = gluon_data.DataLoader(test.transform_first(transform_test), batch_size, shuffle = True, num_workers = workers)

    return train_iter, test_iter