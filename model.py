import os;
import tensorflow as tf;
import numpy as np;

from tqdm import tqdm;
from skimage.io import imsave;
from dataset import *;
from nn import *;

class Model(object):
	def __init__(self,params,phase):
		self.params = params;
		self.phase = phase;
		self.batch_size = params.batch_size if phase=='train' else 1;
		self.batch_norm = params.batch_norm;
		self.image_size = [params.image_width,params.image_height];
		self.image_shape = self.image_size+[1] if params.greyscale else self.image_size+[3];
		self.num_labels = params.num_labels;
		self.capsule_types = params.capsule_types;
		self.num_routing = params.num_routing;
		self.save_dir = os.path.join(params.save_dir,self.params.solver+'/');
		self.global_step = tf.Variable(0,name='global_step',trainable=False);
		self.saver = tf.train.Saver(max_to_keep = 10);
		self.epsilon = 1e-9;
		self.build();

	def build(self):
		bn = self.batch_norm;
		images = tf.placeholder(tf.float32,[self.batch_size]+self.image_shape);
		labels = tf.placeholder(tf.float32,[self.batch_size,self.num_labels]);
		train = tf.placeholder(tf.bool);
		conv1 = convolution(images,9,9,256,1,1,'conv1',padding='VALID');
		conv1 = batch_norm(conv1,'bn1',train,bn);
		conv1 = nonlinear(conv1,'relu');
		
		'''First Capsule layer'''
		primary_capsule = convolution(conv1,9,9,self.capsule_types*8,2,2,'primary_capsule',padding='VALID');
		primary_capsule = tf.reshape(primary_capsule,[self.batch_size,-1,8]);
		primary_capsule = squash(primary_capsule);
		capsule_shape = primary_capsule.get_shape().as_list();

		'''Dynamic routing algorithm'''
		weights = weight('digit_capsule',[1,capsule_shape[1],self.num_labels,8,16],'xavier');
		weights = tf.tile(weights,[self.batch_size,1,1,1,1]);
		capsule_reshape = tf.reshape(primary_capsule,[self.batch_size,-1,1,8,1]);
		capsule_tiled = tf.tile(capsule_reshape,[1,1,self.num_labels,1,1]);
		prediction_vectors = tf.matmul(weights,capsule_tiled,transpose_a=True);
		logits = tf.zeros([1,capsule_shape[1],self.num_labels,1,1]);
		logits = tf.tile(logits,[self.batch_size,1,1,1,1]);
		for i in range(self.num_routing):
			logits = tf.nn.softmax(logits,dim=2);
			digit_capsule = squash(tf.reduce_sum(tf.multiply(logits,prediction_vectors),1,keep_dims=True));
			output_tiled = tf.tile(digit_capsule,[1,capsule_shape[1],1,1,1]);
			update = tf.matmul(prediction_vectors,output_tiled, transpose_a=True);
			logits += tf.reduce_sum(update,axis=0,keep_dims=True);

		digit_capsule = tf.reshape(digit_capsule,[-1,self.num_labels,16]);
		vectors = tf.sqrt(tf.reduce_sum(tf.square(digit_capsule),axis=2,keep_dims=True)+self.epsilon);
		vectors = tf.reshape(vectors,[self.batch_size,self.num_labels]);
		output = tf.nn.softmax(vectors);
		prediction = tf.argmax(output,axis=1);
		prediction = tf.reshape(prediction,[self.batch_size]);
		
		'''Decoder to reconstruct the image'''
		image_size = reduce(lambda x,y: x*y,self.image_shape);
		masked_output = tf.matmul(tf.squeeze(digit_capsule),tf.reshape(labels,[self.batch_size,self.num_labels,1]),transpose_a=True);
		masked_output = tf.reshape(masked_output,[self.batch_size,-1]);
		fc1 = fully_connected(masked_output,512,'fc1');
		fc1 = nonlinear(fc1,'relu');
		fc2 = fully_connected(fc1,1024,'fc2');
		fc2 = nonlinear(fc2,'relu');
		fc3 = fully_connected(fc2,image_size,'fc3');
		decoded = nonlinear(fc3,'sigmoid');
		decoded = tf.reshape(decoded,[self.batch_size]+self.image_shape);

		'''Loss Function'''
		present_loss = tf.square(tf.maximum(0.0,self.params.m_plus-vectors));
		absent_loss = tf.square(tf.maximum(0.0,vectors-self.params.m_minus));
		classifier_loss = labels*(present_loss)+self.params.downweight*(1-labels)*absent_loss;
		classifier_loss = tf.reduce_mean(tf.reduce_sum(classifier_loss,axis=1));
		reconstruct_loss = tf.reduce_mean(tf.square(decoded-images));
		loss = classifier_loss+self.params.regularize_factor*reconstruct_loss;

		if self.params.solver == 'adam':
			solver = tf.train.AdamOptimizer(self.params.learning_rate);
		elif self.params.solver == 'momentum':
			solver = tf.train.MomentumOptimizer(self.params.learning_rate,self.params.momentum);
		elif self.params.solver == 'rmsprop':
			solver = tf.train.RMSPropOptimizer(self.params.learning_rate,self.params.weight_decay,self.params.momentum);
		else:
			solver = tf.train.GradientDescentOptimizer(self.params.learning_rate);

		optimizer = solver.minimize(loss,global_step=self.global_step);

		self.images = images;
		self.labels = labels;
		self.train = train;
		self.primary_capsule = primary_capsule;
		self.digit_capsule = digit_capsule;
		self.prediction = prediction;
		self.decoded = decoded;
		self.capsule_shape = capsule_shape;
		self.classifier_loss = classifier_loss;
		self.reconstruct_loss = reconstruct_loss;
		self.loss = loss;
		self.optimizer = optimizer;

	def Train(self,sess,data):
		print('Training the Model......');
		epochs = self.params.epochs;
		for epoch in tqdm(list(range(epochs)),desc='Epoch'):
			for i in tqdm(list(range(data.batches)),desc='Batch'):
				images,labels = data.next_batch();
				global_step,loss,classifier_loss,reconstruct_loss = sess.run([self.global_step,self.loss,self.classifier_loss,self.reconstruct_loss],feed_dict={self.images:images, self.labels:labels, self.train:True});
				print('Loss = %f Classifier_loss = %f Reconstruction_loss = %f'%(loss,classifier_loss,reconstruct_loss));
				if(global_step%500==0):
					output = sess.run(self.decoded,feed_dict={self.images:images, self.labels:labels});
					self.save_image(output[0],'train_sample_'+str(global_step));
				if(global_step%self.params.save_period==0):
					self.save(sess);
			data.reset();
		self.save(sess);
		print('Model trained......');

	def Test(self,sess,data):
		print('Testing the Model......');
		for i in tqdm(list(range(data.batches)),desc='Batch'):
			images,labels = data.next_batch();
			predictions,accuracy = sess.run([self.predictions,self.accuracy],feed_dict={self.images:images, self.labels:labels, self.train:False});
			print('Batch accuracy = %f'%(accuracy));
		print('Testing completed......');

	def save(self,sess):
		print(('Saving model to %s......'% self.save_dir));
		self.saver.save(sess,self.save_dir,self.generator_step);

	def load(self,sess):
		print('Loading model.....');
		checkpoint = tf.train.get_checkpoint_state(self.save_dir);
		if checkpoint is None:
			print("Error: No saved model found. Please train first...");
			sys.exit(0);
		self.saver.restore(sess, checkpoint.model_checkpoint_path);

	def save_image(self,output,name):
		output = output*255;
		file_name = os.path.join(self.params.train_result,name+'.png');
		imsave(file_name,output);
		print('Saving the image %s...',file_name);