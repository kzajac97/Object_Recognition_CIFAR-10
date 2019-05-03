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

batch_size = 64,  epochs = 32, optimizer = 'sgd', learning_rate = 0.1, weigth_decay = automatic, Loss = SofmaxCrossEntropy
Accuracy = 56.19%

## **Net2**
self.net.add(nn.Conv2D(3,kernel_size=(3,3),padding=1,strides=2,activation='relu'))
self.net.add(nn.BatchNorm())
self.net.add(nn.Conv2D(3,kernel_size=(3,3),padding=1,strides=2,activation='relu'))
self.net.add(nn.BatchNorm())
self.net.add(nn.MaxPool2D())
self.net.add(nn.Dropout(0.2))
self.net.add(nn.Conv2D(3,kernel_size=(3,3),padding=1,strides=2,activation='relu'))
self.net.add(nn.BatchNorm())
self.net.add(nn.Conv2D(3,kernel_size=(3,3),padding=1,strides=2,activation='relu'))
self.net.add(nn.BatchNorm())
self.net.add(nn.MaxPool2D())
self.net.add(nn.Conv2D(3,kernel_size=(3,3),padding=1,strides=2,activation='relu'))
self.net.add(nn.BatchNorm())
self.net.add(nn.Conv2D(3,kernel_size=(3,3),padding=1,strides=2,activation='relu'))
self.net.add(nn.BatchNorm())
self.net.add(nn.MaxPool2D())
self.net.add(nn.Flatten())
self.net.add(nn.Dense(256,activation='softmax'))
self.net.add(nn.Dense(10,activation='softmax'))

self.net.initialize(init.Xavier())

batch_size = 64, epochs = 16, optimizer = 'adam', lr = 0.1, wd = 'auto', Loss = SoftmaxCrossEntropy
Accuracy = 11.91%

## **Net3 **
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
nn.Dense(256,activation='relu')
nn.Dropout(0.2)
nn.Dense(128,activation='relu')
nn.Dropout(0.2)
nn.Dense(64,activation='relu')
nn.Dropout(0.2))
nn.Dense(10,activation='relu')

init.Xavier()

batch_size = 64, epochs = 32, optimizer = 'adam', lr = 0.1, wd = 'auto', Loss = SoftmaxCrossEntropy
Accuracy = 