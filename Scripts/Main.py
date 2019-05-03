import sys
import mxnet as mx
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mxnet import gluon 

from DataReader import *
from Net import *

if __name__ == '__main__':
    # Load CIFAR10 data
    train_data, test_data = Load_CIFAR10()
    # Creates Data Iterators with batch_size 256 using 4 threads
    train_iter, test_iter = Get_Iters(train_data,test_data,256,4)

    if len(sys.argv) == 1:
        deep_net = Recognizer()
        deep_net.load_parameters('recognizer.params')

    else:
        if sys.argv[1] == 'train':  
            deep_net = Recognizer()
         
            loss_function = gluon.loss.SoftmaxCrossEntropyLoss()
            trainer = gluon.Trainer(deep_net.net.collect_params(),'sgd',{'learning_rate' : 0.1} )

            deep_net.train(20, #num_epchos
                            train_iter,    # Data iterator 
                            loss_function, # Softmax
                            trainer,
                            256) #batch_size
          
            deep_net.save_parameters('recognizer.params')

        else:
            print(sys.argv[1])

    print("Testing: ")
    accuracy = 0
    for data,label in test_iter:
        output = deep_net.net(data)
        if output.asnumpy().argmax() == label.asnumpy()[0]:
            accuracy += 1
            
    result = float(accuracy/len(test_data))
    print(result*100,"%")

    sns.regplot(deep_net.loss_values)
    plt.show()