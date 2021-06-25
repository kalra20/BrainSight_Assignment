import numpy as np
# import pandas as pd

import os
import nibabel as nib




from sklearn.model_selection import train_test_split
import tensorflow as tf
# import keras
import tensorflow.keras as keras
from keras import regularizers
from keras import optimizers
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Concatenate, concatenate
from keras.layers import Dense, Dropout, Activation, Flatten, Input, UpSampling2D
from keras.models import Model

# from medpy.io import load
### Commented
# data_path = 'NFBS_Dataset'
# data_dirs = os.listdir(data_path)
##########
def collect_data(data_dirs):
	data_list = []
	for sub_dir in data_dirs:
		dir_contents = os.listdir(os.path.join(data_path,sub_dir))
		filename = sub_dir
		for file in dir_contents:
			if not 'brain' in file:
				image_path =  os.path.join(data_path,sub_dir,file)
			elif 'mask' in file:
				mask_path = os.path.join(data_path,sub_dir,file)
		data_list.append({'filename':filename,'image':image_path,'mask':mask_path})
	return data_list



#########Commented
# data_list = collect_data(data_dirs)


# X = []
# [[X.append(item['image'])] for item in data_list]
# # print(len(X))
# # print(X[0])
# Y = []
# [[Y.append(item['mask'])] for item in data_list]
# print(Y[0])

# #split x and y
# x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.2,random_state=1)
# print(len(x_test),len(y_test))

# trainset = [[xr,yr] for xr,yr in zip(x_train,y_train)]
# # print(k[0][0])
# testset = [[xr,yr] for xr,yr in zip(x_test,y_test)]
# print(trainset[0][1])

############

# from tensorflow.python.keras.utils import data_utils
# keras = tf.compat.v1.keras
# Sequence = keras.utils.Sequence
from tensorflow.keras.utils import Sequence
# from tensorflow.python.keras.utils import data_utils
# class RandomFramesFromPathsToVideos(data_utils.Sequence):
class DatasetGenerator(Sequence):
	def __init__(self, filenames, batch_size=64, crop_dim=[240,240], augment=False, seed=1):
		img = np.array(nib.load(filenames[0][0]).dataobj) # Load the first image
		self.slice_dim = 2  # We'll assume z-dimension (slice) is last
		# Determine the number of slices (we'll assume this is consistent for the other images)
		self.num_slices_per_scan = img.shape[self.slice_dim]  

		self.filenames = filenames
		self.batch_size = batch_size

		self.augment = augment
		self.seed = seed

		self.num_files = len(self.filenames)

		self.ds = self.get_dataset()

	def preprocess_img(self, img):
		"""
		Preprocessing for the image
		z-score normalize
		"""
		return (img - img.mean()) / img.std()

    
	def augment_data(self, img, msk):
		"""
		Data augmentation
		Flip image and mask. Rotate image and mask.
		"""

		if np.random.rand() > 0.5:
		    ax = np.random.choice([0,1])
		    img = np.flip(img, ax)
		    msk = np.flip(msk, ax)

		if np.random.rand() > 0.5:
		    rot = np.random.choice([1, 2, 3])  # 90, 180, or 270 degrees

		    img = np.rot90(img, rot, axes=[0,1])  # Rotate axes 0 and 1
		    msk = np.rot90(msk, rot, axes=[0,1])  # Rotate axes 0 and 1

		return img, msk

	def generate_batch_from_files(self):
	    """
	    Python generator which goes through a list of filenames to load.
	    The files are 3D image (slice is dimension index 2 by default). However,
	    we need to yield them as a batch of 2D slices. This generator
	    keeps yielding a batch of 2D slices at a time until the 3D image is 
	    complete and then moves to the next 3D image in the filenames.
	    An optional `randomize_slices` allows the user to randomize the 3D image 
	    slices after loading if desired.
	    """
	    import nibabel as nib

	    np.random.seed(self.seed)  # Set a random seed

	    idx = 0
	    idy = 0

	    while True:

	        """
	        Pack N_IMAGES files at a time to queue
	        """
	        NUM_QUEUED_IMAGES = 1 + self.batch_size // self.num_slices_per_scan  # Get enough for full batch + 1
	        
	        for idz in range(NUM_QUEUED_IMAGES):

	            label_filename = self.filenames[idx][1]
	            img_filename = self.filenames[idx][0]
	            img = np.array(nib.load(img_filename).dataobj)
	            img = img[:,:,:]  
	            img = self.preprocess_img(img)

	            label = np.array(nib.load(label_filename).dataobj)
	            

	            if idz == 0:
	                img_stack = img
	                label_stack = label

	            else:

	                img_stack = np.concatenate((img_stack,img), axis=self.slice_dim)
	                label_stack = np.concatenate((label_stack,label), axis=self.slice_dim)
	            
	            idx += 1 
	            if idx >= len(self.filenames):
	                idx = 0
	                np.random.shuffle(self.filenames) # Shuffle the filenames for the next iteration
	        
	        img = img_stack
	        label = label_stack

	        num_slices = img.shape[self.slice_dim]
	        
	        if self.batch_size > num_slices:
	            raise Exception("Batch size {} is greater than"
	                            " the number of slices in the image {}."
	                            " Data loader cannot be used.".format(self.batch_size, num_slices))

	        """
	        We can also randomize the slices so that no 2 runs will return the same slice order
	        for a given file. This also helps get slices at the end that would be skipped
	        if the number of slices is not the same as the batch order.
	        """
	        if self.augment:
	            slice_idx = np.random.choice(range(num_slices), num_slices)
	            img = img[:,:,slice_idx]  # Randomize the slices
	            label = label[:,:,slice_idx]

	        name = self.filenames[idx]
	        
	        if (idy + self.batch_size) < num_slices:  # We have enough slices for batch
	            img_batch, label_batch = img[:,:,idy:idy+self.batch_size], label[:,:,idy:idy+self.batch_size]   

	        else:  # We need to pad the batch with slices

	            img_batch, label_batch = img[:,:,-self.batch_size:], label[:,:,-self.batch_size:]  # Get remaining slices

	        if self.augment:
	            img_batch, label_batch = self.augment_data(img_batch, label_batch)
	            
	        if len(np.shape(img_batch)) == 3:
	            img_batch = np.expand_dims(img_batch, axis=-1)
	        if len(np.shape(label_batch)) == 3:
	            label_batch = np.expand_dims(label_batch, axis=-1)
	            
	        yield np.transpose(img_batch, [2,0,1,3]).astype(np.float32), np.transpose(label_batch, [2,0,1,3]).astype(np.float32)


	        idy += self.batch_size
	        if idy >= num_slices: # We finished this file, move to the next
	            idy = 0
	            idx += 1

	        if idx >= len(self.filenames):
	            idx = 0
	            np.random.shuffle(self.filenames) # Shuffle the filenames for the next iteration

	def get_dataset(self):
	    """
	    Return a dataset
	    """
	    ds = self.generate_batch_from_files()
	    
	    return ds  

	def __len__(self):
	    return int((self.num_slices_per_scan * self.num_files)//self.batch_size)

	def __getitem__(self, idx):
	    return next(self.ds)


class SliceGenerator(Sequence):
	def __init__(self, filenames, batch_size=64, crop_dim=[240,240], augment=False, seed=1):
		
		img = np.array(nib.load(filenames).dataobj) # Load the first image
		print(img.shape)
		if img.shape[1] == 256 and img.shape[2] == 256:
			img = np.moveaxis(img,0,2)#img.tranpose(1,2,0)
		elif img.shape[0] == 256 and img.shape[2] == 256:
			img = np.moveaxis(img,1,2)#img.tranpose(0,2,1)
		print(img.shape)
		self.slice_dim = 2  # We'll assume z-dimension (slice) is last
		# Determine the number of slices (we'll assume this is consistent for the other images)
		self.num_slices_per_scan = img.shape[self.slice_dim]  
		self.seed = seed
		self.filenames = filenames
		self.batch_size = img.shape[2]
		print(self.batch_size)
		self.augment = augment
		self.num_files = len(self.filenames)

		self.ds = self.get_dataset()

	def normalize(self, img):
        # this function normalize inputs for zero mean and unit variance
        # it is used when training a model.

		mean = 0.0020320644
		std = 1.0023067
		img = (img - mean) / (std + 1e-7)
		return img

	def generate_batch_from_files(self):
	    """
	    Python generator which goes through a list of filenames to load.
	    The files are 3D image (slice is dimension index 2 by default). However,
	    we need to yield them as a batch of 2D slices. This generator
	    keeps yielding a batch of 2D slices at a time until the 3D image is 
	    complete and then moves to the next 3D image in the filenames.
	    An optional `randomize_slices` allows the user to randomize the 3D image 
	    slices after loading if desired.
	    """
	    import nibabel as nib

	    np.random.seed(self.seed)  # Set a random seed

	    idx = 0
	    idy = 0

	    while True:

	        """
	        Pack N_IMAGES files at a time to queue
	        """
	        NUM_QUEUED_IMAGES = 1 + self.batch_size // self.num_slices_per_scan  # Get enough for full batch + 1
	        
	        for idz in range(NUM_QUEUED_IMAGES):

	            # label_filename = self.filenames[idx][1]
	            img_filename = self.filenames
	            img = np.array(nib.load(img_filename).dataobj)
	            if img.shape[1] == 256 and img.shape[2] == 256:
	            	img = np.moveaxis(img,0,2)
	                # img = np.rollaxis(img,2,0)#img.tranpose(1,2,0)
	            elif img.shape[0] == 256 and img.shape[2] == 256:
	            	img = np.moveaxis(img,1,2)
	                # img = np.rollaxis(img,2,1)#img.tranpose(0,2,1)
	            img = img[:,:,:]  
	            img = self.normalize(img)

	            # label = np.array(nib.load(label_filename).dataobj)
	            

	            if idz == 0:
	                img_stack = img
	                # label_stack = label

	            else:

	                img_stack = np.concatenate((img_stack,img), axis=self.slice_dim)
	                # label_stack = np.concatenate((label_stack,label), axis=self.slice_dim)
	            
	            idx += 1 
	            if idx >= len(self.filenames):
	                idx = 0
	                np.random.shuffle(self.filenames) # Shuffle the filenames for the next iteration
	        
	        img = img_stack
	        # label = label_stack

	        num_slices = img.shape[self.slice_dim]
	        
	        if self.batch_size > num_slices:
	            raise Exception("Batch size {} is greater than"
	                            " the number of slices in the image {}."
	                            " Data loader cannot be used.".format(self.batch_size, num_slices))

	        """
	        We can also randomize the slices so that no 2 runs will return the same slice order
	        for a given file. This also helps get slices at the end that would be skipped
	        if the number of slices is not the same as the batch order.
	        """
	        if self.augment:
	            slice_idx = np.random.choice(range(num_slices), num_slices)
	            img = img[:,:,slice_idx]  # Randomize the slices
	            # label = label[:,:,slice_idx]

	        name = self.filenames[idx]
	        
	        if (idy + self.batch_size) < num_slices:  # We have enough slices for batch
	            img_batch = img[:,:,idy:idy+self.batch_size]   

	        else:  # We need to pad the batch with slices

	            img_batch = img[:,:,-self.batch_size:]  # Get remaining slices

	            
	        if len(np.shape(img_batch)) == 3:
	            img_batch = np.expand_dims(img_batch, axis=-1)
	        # if len(np.shape(label_batch)) == 3:
	        #     label_batch = np.expand_dims(label_batch, axis=-1)
	            
	        yield np.transpose(img_batch, [2,0,1,3]).astype(np.float32)

	        idy += self.batch_size
	        if idy >= num_slices: # We finished this file, move to the next
	            idy = 0
	            idx += 1

	        if idx >= len(self.filenames):
	            idx = 0
	            np.random.shuffle(self.filenames) # Shuffle the filenames for the next iteration

	def get_dataset(self):
	    """
	    Return a dataset
	    """
	    ds = self.generate_batch_from_files()
	    
	    return ds  

	def __len__(self):
	    return int((self.num_slices_per_scan * self.num_files)//self.batch_size)

	def __getitem__(self, idx):
	    return next(self.ds)
##### Commented
# train_generator = DatasetGenerator(trainset,64)
# valid_generator = DatasetGenerator(testset,64)
# print(len(train_generator))
# # lx,ly = train_generator
# # print(lx.shape)
# # print(ly.shape)
# # x_samp = X[0]
# # y_samp = Y[0]
# load_x = []
# load_y = []
# load_x,load_y = valid_generator[0]
# for i in range(1,len(valid_generator)):
# 	print(i)
# 	x_img,y_img = train_generator[i]
# 	load_x = np.vstack((load_x,x_img))
# 	load_y = np.vstack((load_y,y_img))

# print('il',load_x.shape,load_y.shape)
# np.savez_compressed(os.path.join(data_path, 'test.npz'),
#                         x_train=load_x, y_train=load_y)