# Object_Recognition_CIFAR-10

## **Net1**     
nn.Conv2D(3,kernel_size=3,padding=1,strides=1,activation='relu')
nn.BatchNorm()
nn.Conv2D(3,kernel_size=3,padding=1,strides=1,activation='relu')
nn.BatchNorm()
nn.MaxPool2D()
nn.Dropout(0.2)
nn.Conv2D(3,kernel_size=3,padding=1,strides=1,activation='relu')
nn.BatchNorm()
nn.Conv2D(3,kernel_size=3,padding=1,strides=1,activation='relu')
nn.BatchNorm()
nn.MaxPool2D()
nn.Conv2D(3,kernel_size=3,padding=1,strides=1,activation='relu')
nn.BatchNorm()
nn.Conv2D(3,kernel_size=3,padding=1,strides=1,activation='relu')
nn.BatchNorm()
nn.MaxPool2D()
nn.Flatten()
nn.Dense(10,activation='relu')

init.Xavier()

batch_size = 64,  epochs = 100,optimizer = 'sgd', learning_rate = 0.1, weigth_decay = automatic, Loss = SofmaxCrossEntropy
Accuracy = 