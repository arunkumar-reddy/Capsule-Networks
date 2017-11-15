import math;
import numpy as np;
import tensorflow as tf;
from tensorflow.python.training import moving_averages;

'''Define a weight variable'''
def weight(name,shape,init='he',range=0.1,stddev=0.01,init_val=None,group_id=0):
	if init_val is not None:
		initializer = tf.constant_initializer(init_val);
	elif init == 'uniform':
		initializer = tf.random_uniform_initializer(-range,range);
	elif init == 'normal':
		initializer = tf.random_normal_initializer(stddev=stddev);
	elif init == 'he':
		fan_in, _ = get_dimensions(shape);
		std = math.sqrt(2.0/fan_in);
		initializer = tf.random_normal_initializer(stddev=std);
	elif init == 'xavier':
		fan_in,fan_out = get_dimensions(shape);
		range = math.sqrt(6.0/(fan_in+fan_out));
		initializer = tf.random_uniform_initializer(-range,range);
	else:
		initializer = tf.truncated_normal_initializer(stddev=stddev);

	var = tf.get_variable(name,shape,initializer=initializer);
	tf.add_to_collection('l2_'+str(group_id),tf.nn.l2_loss(var));
	return var;

'''Define a bias variable'''
def bias(name,dim,init_val=0.0):
	dimension = dim if isinstance(dim, list) else [dim];
	initializer = tf.constant_initializer(init_val);
	return tf.get_variable(name,dimension,initializer=initializer);

'''Apply a nonlinearity layer''' 
def nonlinear(weights,nonlinearity=None):
	if nonlinearity == 'relu':
		return tf.nn.relu(weights);
	elif nonlinearity == 'tanh':
		return tf.tanh(weights);
	elif nonlinearity == 'sigmoid':
		return tf.sigmoid(weights);
	else:
		return weights;

'''Apply a convolutional layer'''
def convolution(feats,filter_h,filter_w,output_depth,stride_h,stride_w,name,init_w='he',init_b=0,stddev=0.01,padding='SAME',group_id=0):
	input_depth = getshape(feats)[-1];
	convolve = lambda feats,weights: tf.nn.conv2d(feats,weights,strides=[1,stride_h,stride_w,1],padding=padding);
	with tf.variable_scope(name) as scope:
		weights = weight('weights',[filter_h,filter_w,input_depth,output_depth],init=init_w,stddev=stddev,group_id=group_id);
		output = convolve(feats,weights);
		biases = bias('biases',output_depth,init_b);
		output = tf.nn.bias_add(output,biases);
	return output;

'''Apply a convolutional layer without bias'''
def convolution_no_bias(feats,filter_h,filter_w,output_depth,stride_h,stride_w,name,init_w='he',stddev=0.01,padding='SAME',group_id=0):
	input_depth = getshape(feats)[-1];
	convolve = lambda feats,weights: tf.nn.conv2d(feats,weights,strides=[1,stride_h,stride_w,1],padding=padding);
	with tf.variable_scope(name) as scope:
		weights = weight('weights',[filter_h,filter_w,input_depth,output_depth],init=init_w,stddev=stddev,group_id=group_id);
		output = convolve(feats,weights);
	return output;

'''Apply a fully-connected layer'''
def fully_connected(feats,output_size,name,init_w='he',init_b=0,stddev=0.01,group_id=0):
	input_dim = getshape(feats)[-1];
	with tf.variable_scope(name) as scope:
		weights = weight('weights',[input_dim,output_size],init=init_w,stddev=stddev,group_id=group_id);
		biases = bias('biases',[output_size],init_b);
		output = tf.nn.xw_plus_b(feats,weights,biases);
	return output;

'''Apply a fully-connected layer without bias'''
def fully_connected_no_bias(feats,output_size,name,init_w='he',stddev=0.01,group_id=0):
	input_dim = getshape(feats)[-1];
	with tf.variable_scope(name) as scope:
		weights = weight('weights',[input_dim,output_size],init=init_w, stddev=stddev, group_id=group_id);
		output = tf.matmul(feats,weights);
	return output;

'''Apply a batch normalization layer and a nonlinearity layer'''
def batch_norm(weights,name,train,bn=True):
	if bn:
		weights = normalise(weights,name,train);
	return weights;

'''Apply batch normalization'''
def normalise(weights,name,train):
	with tf.variable_scope(name):
		inputs_shape = weights.get_shape();
		axis = list(range(len(inputs_shape) - 1));
		param_shape = int(inputs_shape[-1]);
		moving_mean = tf.get_variable('mean',[param_shape],initializer=tf.constant_initializer(0.0),trainable=False);
		moving_variance = tf.get_variable('variance',[param_shape],initializer=tf.constant_initializer(1.0),trainable=False);
		beta = tf.get_variable('offset',[param_shape],initializer=tf.constant_initializer(0.0));
		gamma = tf.get_variable('scale',[param_shape],initializer=tf.constant_initializer(1.0));
		control_inputs = [];

		def mean_variance_with_update():
			mean,variance = tf.nn.moments(weights, axis);
			update_moving_mean = moving_averages.assign_moving_average(moving_mean,mean,0.99);
			update_moving_variance = moving_averages.assign_moving_average(moving_variance,variance,0.99);
			control_inputs = [update_moving_mean, update_moving_variance];
			return tf.identity(mean),tf.identity(variance);

		def mean_variance():
			mean = moving_mean;
			variance = moving_variance;            
			return tf.identity(mean),tf.identity(variance);

		mean, variance = tf.cond(train,mean_variance_with_update,mean_variance);
		with tf.control_dependencies(control_inputs):
			normed = tf.nn.batch_normalization(weights,mean,variance,beta,gamma,1e-3);
	return normed;

'''Apply a dropout layer'''
def dropout(inputs,keep_prob,train):
	return tf.cond(train,lambda: tf.nn.dropout(inputs,keep_prob),lambda: inputs);

'''Apply a max pooling layer'''
def max_pool(feats,filter_h,filter_w,stride_h,stride_w,name,padding='SAME'):
	return tf.nn.max_pool(feats, ksize=[1,filter_h,filter_w,1], strides=[1,stride_h,stride_w,1],padding=padding,name=name);

'''Apply a average pooling layer'''
def avg_pool(feats,filter_h,filter_w,stride_h,stride_w,name,padding='SAME'):
	return tf.nn.avg_pool(feats, ksize=[1,filter_h,filter_w,1], strides=[1,stride_h,stride_w,1],padding=padding,name=name);

def get_dimensions(shape):
	fan_in = np.prod(shape[:-1])
	fan_out = shape[-1]
	return fan_in,fan_out;

'''Get the shape of a Tensor'''
def getshape(inputs):
	return inputs.get_shape().as_list();

'''Squashing function for capsule layers'''
def squash(inputs,axis=-1):
	epsilon = 1e-9;
	squared_norm = tf.reduce_sum(tf.square(inputs),-1,keep_dims=True);
	scale = squared_norm/(1+squared_norm)/tf.sqrt(squared_norm+epsilon);
	return scale*inputs;