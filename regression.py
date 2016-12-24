'''
Face rotation params(yaw roll pitch) regression with landmarks data as input using TF
Author: YadiraF 
Mail: fengyao@sjtu.edu.cn
Date: 2016/12/24
Ref: https://github.com/aymericdamien/TensorFlow-Examples/blob/master/notebooks/3_NeuralNetworks/multilayer_perceptron.ipynb 
'''
import tensorflow as tf
import numpy as np
import argparse
from sklearn.utils import shuffle

from common import *

## Network Params
n_input=18 # 9 points landmarks
n_hidden_1=256
n_hidden_2=256
n_output= 3 # 3 rotation param

# Store layers weight & bias
weights={
	'h1': tf.Variable(tf.random_normal([n_input,n_hidden_1])),
	'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
	'out': tf.Variable(tf.random_normal([n_hidden_2, n_output]))
}
biases = {
	'b1': tf.Variable(tf.random_normal([n_hidden_1])),
	'b2': tf.Variable(tf.random_normal([n_hidden_2])),
	'out': tf.Variable(tf.random_normal([n_output]))
} 

#Create model
def multilayer_perceptron(x,weight,biases):
	# Hidden layer 
	layer_1=tf.add(tf.matmul(x,weights['h1']),biases['b1'])
	layer_1=tf.nn.sigmoid(layer_1)
	# Hidden layer 
	layer_2=tf.add(tf.matmul(layer_1,weights['h2']),biases['b2'])
	layer_1=tf.nn.sigmoid(layer_2)
	# output layer
	out_layer=tf.matmul(layer_2,weights['out'])+biases['out']
	return out_layer

# tf Graph input
x = tf.placeholder("float", [None, n_input])
y = tf.placeholder("float", [None, n_output])

# Construct model
pred=multilayer_perceptron(x,weights,biases)

# 'Saver' op to save and restore all the variables
saver = tf.train.Saver()

def train(args):
	model_path=args.model_path
	
	# Parameters
	learning_rate=0.03
	training_epoches=5000
	display_step=500
	batch_size=5000

	# Training Data
	X_train=np.loadtxt(args.train_landmarks_path)
	Y_train=np.loadtxt(args.train_rotation_param_path)
	X_train,Y_train=shuffle(X_train,Y_train)
	num_examples=X_train.shape[0]
	
	# Define loss and optimizer
	# mean squared error
	#cost=tf.reduce_sum(tf.pow(pred-y,2))/num_examples
	#cost=tf.reduce_sum(tf.square(tf.sub(pred,y)),1)
	cost=tf.nn.l2_loss(pred-y)
	#cost=tf.reduce_mean(tf.square(pred-y),1)
	optimizer=tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
	#optimizer=tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

	# Initializing the variables
	init=tf.initialize_all_variables()

	# Launch the graph
	with tf.Session() as sess:
		sess.run(init)

		# Training cycle
		for epoch in range(training_epoches):
			avg_cost=0.
			total_batch=int(num_examples/batch_size)
			#total_batch=1
			# Loop over all batches
			for i in range(total_batch):
				batch_x,batch_y=X_train[i*batch_size:(i+1)*batch_size],Y_train[i*batch_size:(i+1)*batch_size]
				#batch_x,batch_y=X_train,Y_train
				# Run optimization op (backprop) and cost op (to get loss value)
				_,c=sess.run([optimizer,cost],feed_dict={x:batch_x,y:batch_y})

				# Comput average loss
				avg_cost +=np.sum(c/total_batch)
			# Display logs per epoch step
			if epoch % display_step==0:
				print "Epoch:",'%04d'%(epoch+1),"cost=","{:.9f}".format(avg_cost)
		print "Optimization Finished!"
		
		# Save model weights to disk
		save_path = saver.save(sess, model_path)
		print "Model saved in file: %s" % save_path

		# Test model
		# Calculate accuracy
		d=3
		print '---------------training data----------------'
		pd=pred.eval({x:X_train[:d,:]})
		print Y_train[:d,:]
		print '----------------'
		print pd[:d,:]
		print '----------------'
		np.set_printoptions(suppress=True)	
		print np.abs(pd-Y_train[:d,:])
		print np.mean((pd-Y_train[:d,:])**2)
		
def test(args):
	model_path=args.model_path	
	
	# Initializing the variables
	init=tf.initialize_all_variables()

	# Test Data
	#X_test=np.loadtxt(args.test_landmarks_path)
	X_test=read_key_all('test-data/data/')
	if args.test_rotation_param_path !='':
		Y_test=np.loadtxt(args.test_rotation_param_path)

	with tf.Session() as sess:
		sess.run(init)

		# restore model weights from saved model
		load_path = saver.restore(sess, model_path)
		print "Model restored from file: %s" % load_path
		
		# output
		output=pred.eval({x:X_test})
		np.savetxt(args.output_rotation_param_path,output)

		# show test result
		d=6
		print '----------------test data---------------'
		print '--predicted answer--'
		print output[:d,:]
		print '----------------'
		print '--- right answer---'
		if args.test_rotation_param_path !='':
			print Y_test[:d,:]
			np.set_printoptions(suppress=True)	
			print '--- differences---'
			print np.abs(output[:d,:]-Y_test[:d,:])
			print np.mean(np.abs(output[:d,:]-Y_test[:d,:]),1)

if __name__=='__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument('--train_flag', type=int, default=0,
						help='if 1, will train the model, default is 0')
	parser.add_argument('--train_landmarks_path', type=str, default="generated-landmarks/landmarks_9_250.txt",
						help='train landmarks path')
	parser.add_argument('--train_rotation_param_path', type=str, default="generated-landmarks/rotation_param_9_250.txt",
						help='train rotation param path')
	parser.add_argument('--test_landmarks_path', type=str, default="generated-landmarks/landmarks_9_100_test.txt",
						help='test landmarks path')
	parser.add_argument('--test_rotation_param_path', type=str, default='',
						help='test rotation param path, for eval')
	parser.add_argument('--output_rotation_param_path', type=str, default='TF-model/output.txt',
						help='output rotation param path, for saving the result')
	parser.add_argument('--model_path', type=str, default='TF-model/model_250.ckpt',
						help='tensorflow model path')
	args = parser.parse_args()
		
	# model path
	if args.train_flag:
		train(args)
	else:
		test(args)
