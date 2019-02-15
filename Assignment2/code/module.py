import mxnet as mxn
from mxnet import nd, autograd, gluon, init
from mxnet.gluon import nn
from time import time
from os.path import join,dirname,abspath

wfolder = join(dirname(dirname(abspath("module.py"))),"weights")
def accuracy(Y_pred,Y_true):
    temp = Y_true == nd.argmax(Y_pred, axis = 1)
    return (100*temp.sum().asscalar()/len(temp))

class Network1(object):

    def __init__(self):
        mxn.random.seed(42)
        self.net = nn.Sequential()
        with self.net.name_scope():
            self.net.add(
                nn.Dense(512,activation = 'relu'),
                nn.Dense(128,activation = 'relu'),
                nn.Dense(64,activation = 'relu'),
                nn.Dense(32,activation = 'relu'),
                nn.Dense(16,activation = 'relu'),
                nn.Dense(10))
        self.net.initialize()
        self.lossfn = gluon.loss.SoftmaxCrossEntropyLoss()
        self.train_loss_his = []
        self.epoch_his = []
        self.val_loss_his = []

    def train(self,X_train,Y_train,X_val,Y_val,epoch = 3,batch_size = 256,lr = 0.001,opt = 'adam'):
        start = time()
        trainer = gluon.Trainer(self.net.collect_params(), opt, {"learning_rate" : lr})

        min_val = 100
        for i in range(epoch):
            train_loss, train_acc, val_acc, val_loss = 0., 0., 0., 0.
            tic = time()

            for j in range(0,len(X_train),batch_size):
                k = min(j+batch_size,len(X_train))
                with autograd.record():
                    output = self.net(X_train[j:k,:])
                    loss = self.lossfn(output,Y_train[j:k])
                loss.backward()
                trainer.step(batch_size)
                train_loss += loss.mean().asscalar()
                train_acc += accuracy(output,Y_train[j:k])

            train_loss = train_loss*batch_size/len(X_train)
            train_acc = train_acc*batch_size/len(X_train)
            val_out = self.net(X_val)
            loss = self.lossfn(val_out,Y_val)
            val_loss = loss.mean().asscalar()
            val_acc = accuracy(val_out,Y_val)
            print("Epoch : %d, Training Loss : %.3f, Training Accuracy : %.2f, Validation Loss : %.3f, Validation Accuracy : %.2f, Time : %.lf sec" %(i+1,train_loss,train_acc,val_loss,val_acc,time()-tic))
            self.train_loss_his.append(train_loss)
            self.epoch_his.append(i+1)
            self.val_loss_his.append(val_loss)
            if val_loss <= min_val:
                min_val = val_loss
                self.net.save_parameters(wfolder+"/network1.params")
        print("\nTotal training time : %.lf sec" %(time()-start))

            
    def test(self,X_test,Y_test):
        self.net.load_parameters(wfolder+"/network1.params")
        output = self.net(X_test)
        loss = self.lossfn(output,Y_test)
        print("Test accuracy of Network 1 : %.2f, Test loss of Network 1 : %.3f" %(accuracy(output,Y_test),loss.mean().asscalar()))
        return


class Model(nn.Block):
    def __init__(self,batchnorm,dropout,**kwargs):
        super(Model,self).__init__(**kwargs)
        self.batchnorm = batchnorm
        self.dropout = dropout
        with self.name_scope():
            self.layer1 = nn.Dense(1024,activation = 'relu')
            if batchnorm:
                self.layer1_batch = nn.BatchNorm(axis=1, center=True, scale=True)
            if dropout != 0:
                self.layer1_drop = nn.Dropout(dropout)
            self.layer2 = nn.Dense(512,activation = 'relu')
            if batchnorm:
                self.layer2_batch = nn.BatchNorm(axis=1, center=True, scale=True)
            if dropout != 0:
                self.layer2_drop = nn.Dropout(dropout)
            self.layer3 = nn.Dense(256,activation = 'relu')
            if batchnorm:
                self.layer3_batch = nn.BatchNorm(axis=1, center=True, scale=True)
            if dropout != 0:
                self.layer3_drop = nn.Dropout(dropout)
            self.layer4 = nn.Dense(10,activation = 'relu')
        return

    def forward(self,x):
        self.h1 = self.layer1(x)
        if self.batchnorm : 
            self.h1 = self.layer1_batch(self.h1)
        if self.dropout != 0:
                self.h1 = self.layer1_drop(self.h1)
        self.h2 = self.layer2(self.h1)
        if self.batchnorm:
            self.h2 = self.layer2_batch(self.h2)
        if self.dropout != 0:
                self.h2 = self.layer1_drop(self.h2)
        self.h3 = self.layer3(self.h2)
        if self.batchnorm:
            self.h3 = self.layer3_batch(self.h3)
        if self.dropout != 0:
                self.h3 = self.layer1_drop(self.h3)
        return self.layer4(self.h3)


class Network2(object):

    def __init__(self,initializer = init.Uniform(scale = 0.07),batchnorm = False,dropout = 0):
        mxn.random.seed(42)
        self.net = Model(batchnorm,dropout)
        self.net.initialize(init = initializer)
        self.lossfn = gluon.loss.SoftmaxCrossEntropyLoss()
        self.train_loss_his = []
        self.epoch_his = []
        self.val_loss_his = []
        

    def train(self,X_train,Y_train,X_val,Y_val,task,epoch = 3,batch_size = 256,lr = 0.001,opt = 'adam'):
        start = time()
        if opt == 'nag':
            trainer = gluon.Trainer(self.net.collect_params(), opt, {"learning_rate" : lr,'momentum' : 0.9})
        else :
            trainer = gluon.Trainer(self.net.collect_params(), opt, {"learning_rate" : lr})

        min_val = 100
        for i in range(epoch):
            train_loss, train_acc, val_acc, val_loss = 0., 0., 0., 0.
            tic = time()

            for j in range(0,len(X_train),batch_size):
                k = min(j+batch_size,len(X_train))
                with autograd.record():
                    output = self.net(X_train[j:k,:])
                    loss = self.lossfn(output,Y_train[j:k])
                loss.backward()
                trainer.step(batch_size)
                train_loss += loss.mean().asscalar()
                train_acc += accuracy(output,Y_train[j:k])

            train_loss = train_loss*batch_size/len(X_train)
            train_acc = train_acc*batch_size/len(X_train)
            val_out = self.net(X_val)
            loss = self.lossfn(val_out,Y_val)
            val_loss = loss.mean().asscalar()
            val_acc = accuracy(val_out,Y_val)
            print("Epoch : %d, Training Loss : %.3f, Training Accuracy : %.2f, Validation Loss : %.3f, Validation Accuracy : %.2f, Time : %.lf sec" %(i+1,train_loss,train_acc,val_loss,val_acc,time()-tic))
            self.train_loss_his.append(train_loss)
            self.epoch_his.append(i+1)
            self.val_loss_his.append(val_loss)
            if val_loss <= min_val:
                min_val = val_loss
                self.net.save_parameters(wfolder+"/network2"+task+".params")
        print("\nTotal training time : %.lf sec" %(time()-start))


    def test(self,X_test,Y_test,task = 'a'):
        self.net.load_parameters(wfolder+"/network2a.params")
        output = self.net(X_test)
        if task[0] == 'c':
            if task[1] == '1':
                return self.net.h1
            if task[1] == '2':
                return self.net.h2
            if task[1] == '3':
                return self.net.h3
        loss = self.lossfn(output,Y_test)
        print("Test accuracy of Network 2 : %.2f, Test loss of Network 2 : %.3f" %(accuracy(output,Y_test),loss.mean().asscalar()))
        return


class logistic_regression(object):
    
    def __init__(self):
        mxn.random.seed(42)
        self.net = nn.Dense(10)
        self.net.collect_params().initialize(init.Normal(sigma=1.))
        self.lossfn = gluon.loss.SoftmaxCrossEntropyLoss()
        self.train_loss_his = []
        self.epoch_his = []
        self.val_loss_his = []
    
    def train(self,X_train,Y_train,X_val,Y_val,task,epoch = 3,batch_size = 256,lr = 0.001,opt = 'adam'):
        start = time()
        trainer = gluon.Trainer(self.net.collect_params(), opt, {"learning_rate" : lr})

        min_val = 100
        for i in range(epoch):
            train_loss, train_acc, val_acc, val_loss = 0., 0., 0., 0.
            tic = time()

            for j in range(0,len(X_train),batch_size):
                k = min(j+batch_size,len(X_train))
                with autograd.record():
                    output = self.net(X_train[j:k,:])
                    loss = self.lossfn(output,Y_train[j:k])
                loss.backward()
                trainer.step(batch_size)
                train_loss += loss.mean().asscalar()
                train_acc += accuracy(output,Y_train[j:k])

            train_loss = train_loss*batch_size/len(X_train)
            train_acc = train_acc*batch_size/len(X_train)
            val_out = self.net(X_val)
            loss = self.lossfn(val_out,Y_val)
            val_loss = loss.mean().asscalar()
            val_acc = accuracy(val_out,Y_val)
            print("Epoch : %d, Training Loss : %.3f, Training Accuracy : %.2f, Validation Loss : %.3f, Validation Accuracy : %.2f, Time : %.lf sec" %(i+1,train_loss,train_acc,val_loss,val_acc,time()-tic))
            self.train_loss_his.append(train_loss)
            self.epoch_his.append(i+1)
            self.val_loss_his.append(val_loss)
            if val_loss <= min_val:
                min_val = val_loss
                self.net.save_parameters(wfolder+"/log_reg"+task+".params")
        print("\nTotal training time : %.lf sec" %(time()-start))

    def test(self,X_test,Y_test,task):
        self.net.load_parameters(wfolder+"/log_reg"+task+".params")
        output = self.net(X_test)
        loss = self.lossfn(output,Y_test)
        print("Test accuracy : %.2f, Test loss : %.3f" %(accuracy(output,Y_test),loss.mean().asscalar()))
        return

