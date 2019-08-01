#import gym
#from RL_brain import DeepQNetwork

import tensorflow as tf
import numpy as np

class tandem_network(object):
	def __init__(self, 
		INN_size,
		FNN_size,
		starter_learning_rate = 0.001,#0.0001
		decay_step = 1000*100,#3000000
		decay_rate = 0.5,
		#learning_rate = 0.05,
		range_response = [-2, 2],
		range_design = [0, 1],
		training=True
		):
		assert INN_size[0]  == FNN_size[-1]
		assert INN_size[-1] == FNN_size[0]
		#super(Deep_network, self).__init__()
		self.global_step = tf.Variable(0, trainable=False)
		self.lr = tf.train.exponential_decay(starter_learning_rate, self.global_step, decay_step, decay_rate, staircase=True)
		self.inn_size = INN_size
		self.fnn_size = FNN_size
		self._train_phase = training#tf.placeholder(tf.bool, name='training_phase') # indicate training or not, used for dropout and BN phase
		self.range_response = range_response
		self.range_design = range_design
		self.build_net()
		self.saver = tf.train.Saver()
		#var_list = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
		#var_list += tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
		#self.saver = tf.train.Saver(var_list)
		
		#self.correct_pred = tf.equal(tf.round(self.pred), self.label)
		#self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred,"float"))
		self.sess = tf.Session()
		self.sess.run(tf.global_variables_initializer())

	def build_net(self):
		self.response = tf.placeholder(tf.float32, [None, self.inn_size[0]],  name='response')  # input
		self.design   = tf.placeholder(tf.float32, [None, self.inn_size[-1]], name='design')  # label
		self.pred_fnn = self.build_dense(self.design, 'FNN', self.fnn_size, self.range_response)
		self.layer_m  = self.build_dense(self.response, 'INN', self.inn_size, self.range_design)
		self.pred_td  = self.build_dense(self.layer_m, 'cpFNN', self.fnn_size, self.range_response)
		cp_params  = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='cpFNN')
		fnn_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='FNN')
		inn_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='INN')
		self.cp_fnn_op = [tf.assign(cp, f) for cp, f in zip(cp_params, fnn_params)]
		self.saver_cpFNN = tf.train.Saver(var_list=cp_params)
		self.saver_FNN = tf.train.Saver(var_list=fnn_params)
		self.saver_INN = tf.train.Saver(var_list=inn_params)
		with tf.variable_scope('loss'):
			#cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction), reduction_indices=[1]))  # cross_entropy loss
			self.loss_fnn = tf.nn.l2_loss(self.response - self.pred_fnn, name='loss_FNN')
			self.loss_inn = tf.nn.l2_loss(self.design   - self.layer_m,  name='loss_INN')
			self.loss_td  = tf.nn.l2_loss(self.response - self.pred_td,  name='loss_tandem')
		with tf.variable_scope('train'):
			self._train_fnn = tf.train.AdamOptimizer(self.lr).minimize(self.loss_fnn, var_list=fnn_params, global_step=self.global_step)
			self._train_inn = tf.train.AdamOptimizer(self.lr).minimize(self.loss_inn, var_list=inn_params, global_step=self.global_step)
			self._train_td  = tf.train.AdamOptimizer(self.lr).minimize(self.loss_td,  var_list=inn_params, global_step=self.global_step)
	
	# build with one name output
	def build_dense(self, input_layer, name, size, output_range=[0,1]):
		w_initializer = tf.random_uniform_initializer(-0.005, 0.005)#FNN
		w_initializer = tf.random_uniform_initializer(-0.01, 0.01)#INN
		regularizer = tf.contrib.layers.l2_regularizer(0.01, scope=None)
		
		with tf.variable_scope(name):
			output = input_layer
			for j in range(len(size)-2):
				output = tf.layers.dense(output, size[j+1], kernel_regularizer=regularizer,	bias_regularizer=regularizer, kernel_initializer=w_initializer, bias_initializer=w_initializer,
					#name='dense_layer'+str(j), 
					#activation = tf.nn.relu
					)
				#output = tf.layers.batch_normalization(output, momentum=0.4, training=self._train_phase, beta_regularizer=regularizer, gamma_regularizer=regularizer,)
				output = tf.nn.relu(output)
				#output = tf.layers.dropout(output, training = self._train_phase)
			pred = tf.layers.dense(output, size[-1], kernel_regularizer=regularizer, bias_regularizer=regularizer, kernel_initializer=w_initializer, bias_initializer=w_initializer,
					#name='dense_layer'+str(j), 
					activation = tf.nn.sigmoid
					)
		return (output_range[1]-output_range[0])*pred + output_range[0]

	def reset_global_step(self):
		self.sess.run(self.global_step.assign(0))

	def copy_FNN(self):
		self.sess.run(self.cp_fnn_op)

	def train(self, design, response, mode):
		if mode=='FNN':
			self.sess.run(self._train_fnn, feed_dict = {self.design: design, self.response: response})#, self._train_phase: True})
		elif mode=='INN':
			self.sess.run(self._train_inn, feed_dict = {self.design: design, self.response: response})#, self._train_phase: True})
		elif mode=='tandem':
			self.sess.run(self._train_td,  feed_dict = {self.design: design, self.response: response})#, self._train_phase: True})
		else:
			raise ValueError('mode should be FNN, INN or tandem.')

	def show_loss(self, design, response, mode):
		if mode=='FNN':
			return self.sess.run(self.loss_fnn, feed_dict = {self.design: design, self.response: response})/len(design)
		elif mode=='INN':
			return self.sess.run(self.loss_inn, feed_dict = {self.design: design, self.response: response})/len(design)
		elif mode=='tandem':
			return self.sess.run(self.loss_td,  feed_dict = {self.design: design, self.response: response})/len(design)
		else:
			raise ValueError('mode should be FNN, INN or tandem.')

	def show_lr(self):
		return self.sess.run(self.lr)

	def test(self, design, response, mode):
		if mode=='FNN':
			return self.sess.run(self.pred_fnn, feed_dict = {self.design: design, self.response: response})#, self._train_phase: False})
		elif mode=='INN':
			return self.sess.run(self.layer_m,  feed_dict = {self.design: design, self.response: response})#, self._train_phase: False})
		elif mode=='tandem':
			return self.sess.run(self.pred_td,  feed_dict = {self.design: design, self.response: response})#, self._train_phase: False})
		else:
			raise ValueError('mode should be FNN, INN or tandem.')

	def save(self, filename):
		self.saver.save(self.sess, filename)
	def restore(self, filename):
		self.saver.restore(self.sess, filename)
	def save_FNN(self, filename):
		self.saver_FNN.save(self.sess, filename)
	def restore_FNN(self, filename):
		self.saver_FNN.restore(self.sess, filename)
		self.copy_FNN()

