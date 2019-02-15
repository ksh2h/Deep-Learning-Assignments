import argparse
import numpy as np
import data_loader
import matplotlib.pyplot as plt
from module import *

parser = argparse.ArgumentParser()
parser.add_argument("--train", help="trains both the networks on training-set", action = "store_true")
parser.add_argument("--test", help="tests and compares the performance of two networks on test-set", action = "store_true")
args = parser.parse_args()
np.random.seed(42)

def shuffle_dataset(X,Y):
	assert len(X) == len(Y)
	rand_state = np.random.get_state()
	np.random.shuffle(X)
	np.random.set_state(rand_state)
	np.random.shuffle(Y)
	return

if args.train :
	print("\nTraining both the networks\n")
	data_obj = data_loader.DataLoader()
	X,Y = data_obj.load_data('train')
	shuffle_dataset(X,Y)
	X_train = X[:42000]
	Y_train = Y[:42000]
	X_val = X[42000:]
	Y_val = Y[42000:]
	net1 = Network1()
	print("\n###################Training Process for Network 1 ########################\n")
	net1.train(nd.array(X_train),nd.array(Y_train),nd.array(X_val),nd.array(Y_val))
	net2 = Network2()
	print("\n###################Training Process for Network 2 ########################\n")
	net2.train(nd.array(X_train),nd.array(Y_train),nd.array(X_val),nd.array(Y_val),'a')
	plt.plot(net1.epoch_his,net1.train_loss_his,label = 'Net1 Training Loss')
	plt.plot(net1.epoch_his,net1.val_loss_his,label = 'Net1 Validation Loss')
	plt.plot(net2.epoch_his,net2.train_loss_his,label = 'Net2 Training Loss')
	plt.plot(net2.epoch_his,net2.val_loss_his,label = 'Net2 Validation Loss')
	plt.xlabel("Epoch")
	plt.ylabel("Training Loss")
	plt.legend()
	plt.show()

if args.test :
	print("\nTesting both the networks\n")
	data_obj = data_loader.DataLoader()
	X,Y = data_obj.load_data('test')
	net1 = Network1()
	print("\n###################Testing Process for Network 1 ########################\n")
	net1.test(nd.array(X),nd.array(Y))
	net2 = Network2()
	print("\n###################Testing Process for Network 2 ########################\n")
	net2.test(nd.array(X),nd.array(Y))

if args.train == False and args.test == False :
	print("Please specify one of the following arguments : \n"+\
		   "1. 'python 14EE35011_Assignment2_task_a.py --train' for training\n"+\
		   "2. 'python 14EE35011_Assignment2_task_a.py --test' for testing\n"+\
		   "3. 'python 14EE35011_Assignment2_task_a.py --train --test' for doing both")


