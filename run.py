"""
"Training Deep Neural Networks for the Inverse Design of Nanophotonic Structures", ACS Photonics 5, 4, 1365-1369 (2018).
Author: Dianjing Liu, Yixuan Tan, Erfan Khoram and Zongfu Yu

This is the implementation of the tandem neural network introduced in the article. 
To reproduce the 2D deisgn result, run the code with mode='FNN' to train the forward network. Then run the code again with mode='tandem'.
"""

import numpy as np
import DNN_tandem

# Select training mode
#mode = 'FNN' # Train forward neural network
mode = 'tandem' # Train inverse part of the tandem network

# network parameters
n_input   = 9 #12
n_classes = 6 #3 
fnn_size  = [n_input, 1024, 512,512,256, 256,128, n_classes]
inn_size  = [n_classes, 512, 256, n_input] 

# load data
trainfile = './data/inverse2d_train.npy'
data_train = np.load(trainfile)

testfile = './data/inverse2d_test.npy'
data_test = np.load(testfile)

# Preprocessing
data_train = np.delete(data_train,[9,10,11],axis=1)
data_test  = np.delete(data_test,[9,10,11],axis=1)
test_x = data_test[:, 0:n_input]
test_y = data_test[:, n_input:n_classes+n_input]

# Set training parameters
# For forward network.
if mode == 'FNN':
	batch_size      = 2000
	training_epochs = 5000
	start_lr        = 0.0005 # learning rate
	decay_rate      = 0.93 # learning rate decay rate
	decay_step      = 400*300 # learning rate decay steps
# For tandem network
elif mode == 'tandem':
	batch_size      = 800
	training_epochs = 2000
	start_lr        = 0.0001
	decay_rate      = 0.93
	decay_step      = 400*100

# Build network
tandem = DNN_tandem.tandem_network(INN_size=inn_size, FNN_size=fnn_size, starter_learning_rate = start_lr,decay_step = decay_step,decay_rate=decay_rate)

# Restore model
if mode == 'FNN':
	tandem.restore_FNN('./model/DNN_tandem_FNN.ckpt')
#tandem.restore('./model/DNN_tandem.ckpt')

tandem.reset_global_step()
import time
t=time.time()
output = open("DNN_"+mode+"-t.txt","w+")

# Training
for n_epoch in range(training_epochs):
	np.random.shuffle(data_train)
	#  mini-batch training
	for i in range(len(data_train)/batch_size):
		batch_x = data_train[i * batch_size : i * batch_size + batch_size, 0 : n_input]
		batch_y = data_train[i * batch_size : i * batch_size + batch_size, n_input : n_classes+n_input]
		tandem.train(batch_x, batch_y, mode)
	# print and save errors
	train_err = tandem.show_loss(batch_x,batch_y,mode)
	test_err  = tandem.show_loss(test_x,test_y,mode)
	lr = tandem.show_lr()
	print(n_epoch, train_err, test_err, lr)
	#for debug
	#print(tandem.test(test_x, test_y, "FNN"))

	output.write(str(train_err)+ ' ' +str(test_err) + ' ' +str(lr)+"\n")


	# save every 100 epochs
	if (n_epoch+1) % 100 == 0:
		if mode == 'FNN':
			tandem.copy_FNN()
			tandem.save_FNN('./model/DNN_tandem_FNN.ckpt')
		elif mode == 'tandem':
			tandem.save('./model/DNN_tandem.ckpt')

	
output.write('%training time: ' + str(time.time()-t))
output.close()
print('training time: ' + str(time.time()-t))



"""
# Testing network
def err(a,b):
	return np.sqrt(np.mean(np.square(a - b)))

tandem = DNN_tandem.tandem_network(INN_size=inn_size, FNN_size=fnn_size, training=True)
#tandem.restore_FNN('./model/DNN_tandem_FNN.ckpt')
tandem.restore('./model/DNN_tandem.ckpt')
# test FNN
response = tandem.test(test_x, test_y, "FNN")
err_rms = err(response,test_y)
print(np.concatenate((test_y, response),axis=1)[0:5])
print(err_rms)

# test INN
design = tandem.test(test_x, test_y, 'INN')
response = tandem.test(design, test_y, 'FNN')
err_rms = err(response,test_y)
print(np.concatenate((test_y, response),axis=1)[0:5])
print(err_rms)
import scipy.io
s = response[:,0:3]
c = response[:,3:6]
scipy.io.savemat('prediction.mat', {'pred_design':design, 'pred_siny':s, 'pred_cosy':c})

'''
# test INN+cpFNN
response_td = tandem.test(test_x, test_y, 'tandem')
err_rms = err(response_td,test_y)
print(np.concatenate((test_y, response_td),axis=1)[0:5])
print(err_rms)
print(tandem.show_loss(test_x,test_y,'tandem'))
'''
"""