import sys
import mxnet as mx
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mxnet import gluon 

from DataReader import *
from Net import *
from ResNet import *

if __name__ == '__main__':
    # Load CIFAR10 data
    train_data, test_data = Load_CIFAR10()
    # Creates Data Iterators with batch_size 256 using 4 threads
    train_iter, test_iter = Get_Iters(train_data,test_data,128,4)      
    deep_net = Resnet()
    
    loss_function = gluon.loss.SoftmaxCrossEntropyLoss()
    trainer = gluon.Trainer(deep_net.net.collect_params(),'sgd',{'learning_rate' : 0.1, 'momentum' : 0.9, 'wd' : 5e-4})

    deep_net.train(20, #num_epchos
                    train_iter,    # Data iterator 
                    loss_function, # Softmax
                    trainer,
                    128) #batch_size
    
    deep_net.save_parameters('recognizer.params')

    print("Testing: ")
    accuracy = 0
    for data,label in test_iter:
        output = deep_net.net(data.as_in_context(deep_net.ctx))
        if output.asnumpy().argmax() == label.asnumpy()[0]:
            accuracy += 1
            
    result = float(accuracy/len(test_data))
    print(result*100,"%")