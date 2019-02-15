import numpy as np
import data_loader
import matplotlib.pyplot as plt
from module import *


np.random.seed(42)
def shuffle_dataset(X,Y):
	assert len(X) == len(Y)
	rand_state = np.random.get_state()
	np.random.shuffle(X)
	np.random.set_state(rand_state)
	np.random.shuffle(Y)
	return


data_obj = data_loader.DataLoader()
X,Y = data_obj.load_data('train')
shuffle_dataset(X,Y)
X_train = X[:42000]
Y_train = Y[:42000]
X_val = X[42000:]
Y_val = Y[42000:]
print("\n######################### Naive network 2 ###############################\n")
net = Network2()
net.train(nd.array(X_train),nd.array(Y_train),nd.array(X_val),nd.array(Y_val),'b')



print("\n######################### Experiment 1 ###############################\n")
net1_a = Network2(initializer = init.Normal(sigma = 0.01))
net1_a.train(nd.array(X_train),nd.array(Y_train),nd.array(X_val),nd.array(Y_val),'b11')
net1_b = Network2(initializer = init.Xavier())
net1_b.train(nd.array(X_train),nd.array(Y_train),nd.array(X_val),nd.array(Y_val),'b12')
net1_c = Network2(initializer = init.Orthogonal(scale = 1.414))
net1_c.train(nd.array(X_train),nd.array(Y_train),nd.array(X_val),nd.array(Y_val),'b13')
plt.plot(net1_a.epoch_his,net1_a.train_loss_his,label = '(Normal init)Net2 Training Loss')
plt.plot(net1_b.epoch_his,net1_b.train_loss_his,label = '(Xavier init)Net2 Training Loss')
plt.plot(net1_c.epoch_his,net1_c.train_loss_his,label = '(Orthogonal init)Net2 Training Loss')
plt.plot(net.epoch_his,net.train_loss_his,label = 'Vanilla(Uniform init) Net2 Training Loss')
plt.xlabel("Epoch")
plt.ylabel("Training Loss")
plt.legend()
plt.show()




print("\n######################### Experiment 2 ###############################\n")
net2 = Network2(batchnorm = True)
net2.train(nd.array(X_train),nd.array(Y_train),nd.array(X_val),nd.array(Y_val),'b2')
plt.plot(net2.epoch_his,net2.train_loss_his,label = '(with batchnorm)Net2 Training Loss')
plt.plot(net.epoch_his,net.train_loss_his,label = 'Vanilla Net2 Training Loss')
plt.xlabel("Epoch")
plt.ylabel("Training Loss")
plt.legend()
plt.show()



print("\n######################### Experiment 3 ###############################\n")
net3_a = Network2(dropout = 0.1)
net3_a.train(nd.array(X_train),nd.array(Y_train),nd.array(X_val),nd.array(Y_val),'b31')
net3_b = Network2(dropout = 0.4)
net3_b.train(nd.array(X_train),nd.array(Y_train),nd.array(X_val),nd.array(Y_val),'b32')
net3_c = Network2(dropout = 0.6)
net3_c.train(nd.array(X_train),nd.array(Y_train),nd.array(X_val),nd.array(Y_val),'b33')
plt.plot(net.epoch_his,net.train_loss_his,label = 'Vanilla Net2 Training Loss')
plt.plot(net3_a.epoch_his,net3_a.train_loss_his,label = '(Dropout = 0.1) Net2 Training Loss')
plt.plot(net3_b.epoch_his,net3_b.train_loss_his,label = '(Dropout = 0.4) Net2 Training Loss')
plt.plot(net3_c.epoch_his,net3_c.train_loss_his,label = '(Dropout = 0.6) Net2 Training Loss')
plt.xlabel("Epoch")
plt.ylabel("Training Loss")
plt.legend()
plt.show()


print("\n######################### Experiment 4 ###############################\n")
net4_a = Network2()
net4_a.train(nd.array(X_train),nd.array(Y_train),nd.array(X_val),nd.array(Y_val),'b41',opt = 'sgd')
net4_b = Network2()
net4_b.train(nd.array(X_train),nd.array(Y_train),nd.array(X_val),nd.array(Y_val),'b42',opt = 'nag')
net4_c = Network2()
net4_c.train(nd.array(X_train),nd.array(Y_train),nd.array(X_val),nd.array(Y_val),'b43',opt = 'adagrad')
net4_d = Network2()
net4_d.train(nd.array(X_train),nd.array(Y_train),nd.array(X_val),nd.array(Y_val),'b44',opt = 'adadelta')
net4_e = Network2()
net4_e.train(nd.array(X_train),nd.array(Y_train),nd.array(X_val),nd.array(Y_val),'b45',opt = 'rmsprop')
plt.plot(net.epoch_his,net.train_loss_his,label = '(ADAM optiimzer) Net2 Training Loss')
plt.plot(net4_a.epoch_his,net4_a.train_loss_his,label = '(SGD optimizer) Net2 Training Loss')
plt.plot(net4_b.epoch_his,net4_b.train_loss_his,label = "(SGD with Nesterov's momentum) Net2 Training Loss")
plt.plot(net4_c.epoch_his,net4_c.train_loss_his,label = '(AdaGrad optimizer) Net2 Training Loss')
plt.plot(net4_d.epoch_his,net4_d.train_loss_his,label = '(AdaDelta optimizer) Net2 Training Loss')
plt.plot(net4_e.epoch_his,net4_e.train_loss_his,label = '(RMSProp optimizer) Net2 Training Loss')
plt.xlabel("Epoch")
plt.ylabel("Training Loss")
plt.legend()
plt.show()
