import os;
import math;
import numpy as np;

class Dataset():
	def __init__(self,images,labels,batch_size):
		self.images = images;
		self.labels = labels;
		self.count = len(images);
		self.batch_size = batch_size;
		self.batches = int(self.count*1.0/self.batch_size);
		self.index = 0;
		self.indices = list(range(self.count));
		print('Dataset built......');

	def reset(self):
		self.index = 0
		np.random.shuffle(self.indices);

	def next_batch(self):
		if(self.index+self.batch_size<=self.count):
			start = self.index;
			end = self.index+self.batch_size;
			current = self.indices[start:end];
			images = self.images[current];
			labels = self.labels[current];
			self.index += self.batch_size;
			return images,labels;

def train_data(args):
	image_file = os.path.join(args.train_dir,'train_images.npy');
	label_file = os.path.join(args.test_dir,'train_labels.npy');
	batch_size = args.batch_size;
	images = np.load(image_file);
	labels = np.load(label_file);
	dataset = Dataset(images,labels,batch_size);
	return dataset;

def test_data(args):
	image_file = os.path.join(args.train_dir,'test_images.npy');
	label_file = os.path.join(args.test_dir,'test_labels.npy');
	batch_size = args.batch_size;
	images = np.load(image_file);
	labels = np.load(label_file);
	dataset = Dataset(images,labels,batch_size);
	return dataset;