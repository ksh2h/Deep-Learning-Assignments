import numpy as np
import matplotlib.pyplot as plt
import data_loader
from module import *

np.random.seed(42)
def shuffle_dataset(X,Y):
	assert len(X) == len(Y)
	rand_state = np.random.get_state()
	np.random.shuffle(X)
	np.random.set_state(rand_state)
	np.random.shuffle(Y)
	return

def accuracy(Y_pred,Y_true):
    temp = Y_true == nd.argmax(Y_pred, axis = 1)
    return (100*temp.sum().asscalar()/len(temp))


data_obj = data_loader.DataLoader()
X,Y = data_obj.load_data('train')
shuffle_dataset(X,Y)
X_train = X[:42000]
Y_train = Y[:42000]
X_val = X[42000:]
Y_val = Y[42000:]
X_test,Y_test = data_obj.load_data('test')
net = Network2()
print("\n########################### Neural Network 2 ############################\n")
net.test(nd.array(X_test),nd.array(Y_test))


print("\n########################### First hidden layer as feauture input to logistic regression ############################\n")
X_train1 = net.test(nd.array(X_train),nd.array(Y_train),'c1')
X_val1 = net.test(nd.array(X_val),nd.array(Y_val),'c1')
X_test1 = net.test(nd.array(X_test),nd.array(Y_test),'c1')
net1 = logistic_regression()
net1.train(nd.array(X_train1),nd.array(Y_train),nd.array(X_val1),nd.array(Y_val),'c1')
net1.test(nd.array(X_test1),nd.array(Y_test),'c1')



print("\n########################### Second hidden layer as feauture input to logistic regression ############################\n")
X_train2 = net.test(nd.array(X_train),nd.array(Y_train),'c2')
X_val2 = net.test(nd.array(X_val),nd.array(Y_val),'c2')
X_test2 = net.test(nd.array(X_test),nd.array(Y_test),'c2')
net2 = logistic_regression()
net2.train(nd.array(X_train2),nd.array(Y_train),nd.array(X_val2),nd.array(Y_val),'c2')
net2.test(nd.array(X_test2),nd.array(Y_test),'c2')


print("\n########################### Third hidden layer as feauture input to logistic regression ############################\n")
X_train3 = net.test(nd.array(X_train),nd.array(Y_train),'c3')
X_val3 = net.test(nd.array(X_val),nd.array(Y_val),'c3')
X_test3 = net.test(nd.array(X_test),nd.array(Y_test),'c3')
net3 = logistic_regression()
net3.train(nd.array(X_train3),nd.array(Y_train),nd.array(X_val3),nd.array(Y_val),'c3')
net3.test(nd.array(X_test3),nd.array(Y_test),'c3')

