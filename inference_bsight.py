from bsight import SliceGenerator
import numpy as np
# import pandas as pd

import os
import nibabel as nib


from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
from sklearn.model_selection import train_test_split
import tensorflow as tf
# import keras
import tensorflow.keras as keras
from keras import regularizers
from keras import optimizers
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Concatenate, concatenate
from keras.layers import Dense, Dropout, Activation, Flatten, Input, UpSampling2D
from keras.models import Model
class CIFARUnet:
    def __init__(self, train=True, filename = 'abcd',maxepochs = 250):
        # self.alpha = alpha
        self.num_classes = 2
        self.weight_decay= 1e-4
        self.weight_decay_fc= 1e-7
        self.weight_decay_rc= 1e-7
        # self.noise = noise_frac
        self.batch_size=16
        # self._load_data()
        # self.gamma = gamma
        # self.d = cost_rej
        # self.x_shape = self.x_train.shape[1:]
        self.filename = filename
        self.maxepochs = maxepochs
        
        # self.train_generator
        # self.valid_generator
        self.model = self.build_model()
        print(self.model.summary())
        if train:
            self.model = self.train(self.model)
        else:

            self.model.load_weights("checkpoints/{}".format(self.filename))
            print('model weights uploaded')
        

    def build_model(self):
        # Build the network of vgg for 10 classes with massive dropout and weight decay as described in the paper.
        acti = 'relu'
        # weight_decay = self.weight_decay
        weight_decay = self.weight_decay
        weight_decay_fc = self.weight_decay_fc
        weight_decay_rc = self.weight_decay_rc
        basic_dropout_rate = 0.3
        n_classes = 1
        # elif IMAGE_ORDERING == 'channels_last':
        MERGE_AXIS = -1
        IMAGE_ORDERING = 'channels_last'
        img_input = Input(shape=(256,256,1))
        conv1 = Conv2D(64, 3, activation = 'relu', kernel_regularizer=regularizers.l2(weight_decay), padding = 'same', kernel_initializer = 'he_normal')(img_input)
        conv1 = BatchNormalization()(conv1)
        drop1 = Dropout(0.3)(conv1)
        conv1 = Conv2D(64, 3, activation = 'relu', kernel_regularizer=regularizers.l2(weight_decay),padding = 'same', kernel_initializer = 'he_normal')(conv1)
        conv1 = BatchNormalization()(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
        conv2 = Conv2D(128, 3, activation = 'relu', kernel_regularizer=regularizers.l2(weight_decay),padding = 'same', kernel_initializer = 'he_normal')(pool1)
        conv2 = BatchNormalization()(conv2)
        drop2 = Dropout(0.3)(conv2)
        conv2 = Conv2D(128, 3, activation = 'relu', kernel_regularizer=regularizers.l2(weight_decay),padding = 'same', kernel_initializer = 'he_normal')(conv2)
        conv2 = BatchNormalization()(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
        conv3 = Conv2D(256, 3, activation = 'relu', kernel_regularizer=regularizers.l2(weight_decay),padding = 'same', kernel_initializer = 'he_normal')(pool2)
        conv3 = BatchNormalization()(conv3)
        drop3 = Dropout(0.3)(conv3)
        conv3 = Conv2D(256, 3, activation = 'relu', kernel_regularizer=regularizers.l2(weight_decay),padding = 'same', kernel_initializer = 'he_normal')(conv3)
        conv3 = BatchNormalization()(conv3)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
        conv4 = Conv2D(512, 3, activation = 'relu', kernel_regularizer=regularizers.l2(weight_decay),padding = 'same', kernel_initializer = 'he_normal')(pool3)
        conv4 = BatchNormalization()(conv4)
        drop4 = Dropout(0.4)(conv4)
        conv4 = Conv2D(512, 3, activation = 'relu', kernel_regularizer=regularizers.l2(weight_decay),padding = 'same', kernel_initializer = 'he_normal')(conv4)
        drop4 = Dropout(0.5)(conv4)
        pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

        conv5 = Conv2D(1024, 3, activation = 'relu', kernel_regularizer=regularizers.l2(weight_decay),padding = 'same', kernel_initializer = 'he_normal')(pool4)
        conv5 = BatchNormalization()(conv5)
        conv5 = Conv2D(1024, 3, activation = 'relu', kernel_regularizer=regularizers.l2(weight_decay),padding = 'same', kernel_initializer = 'he_normal')(conv5)
        conv5 = BatchNormalization()(conv5)
        drop5 = Dropout(0.5)(conv5)

        up6 = Conv2D(512, 2, activation = 'relu', kernel_regularizer=regularizers.l2(weight_decay),padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
        merge6 = concatenate([drop4,up6], axis = 3)
        conv6 = Conv2D(512, 3, activation = 'relu', kernel_regularizer=regularizers.l2(weight_decay),padding = 'same', kernel_initializer = 'he_normal')(merge6)
        # conv6 = BatchNormalization()(conv6)
        conv6 = Conv2D(512, 3, activation = 'relu', kernel_regularizer=regularizers.l2(weight_decay),padding = 'same', kernel_initializer = 'he_normal')(conv6)
        # conv6 = BatchNormalization()(conv6)
        up7 = Conv2D(256, 2, activation = 'relu', kernel_regularizer=regularizers.l2(weight_decay),padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
        merge7 = concatenate([conv3,up7], axis = 3)
        conv7 = Conv2D(256, 3, activation = 'relu', kernel_regularizer=regularizers.l2(weight_decay),padding = 'same', kernel_initializer = 'he_normal')(merge7)
        # conv7 = BatchNormalization()(conv7)
        conv7 = Conv2D(256, 3, activation = 'relu', kernel_regularizer=regularizers.l2(weight_decay),padding = 'same', kernel_initializer = 'he_normal')(conv7)
        # conv7 = BatchNormalization()(conv7)
        up8 = Conv2D(128, 2, activation = 'relu', kernel_regularizer=regularizers.l2(weight_decay),padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
        merge8 = concatenate([conv2,up8], axis = 3)
        conv8 = Conv2D(128, 3, activation = 'relu', kernel_regularizer=regularizers.l2(weight_decay),padding = 'same', kernel_initializer = 'he_normal')(merge8)
        # conv8 = BatchNormalization()(conv8)
        conv8 = Conv2D(128, 3, activation = 'relu', kernel_regularizer=regularizers.l2(weight_decay),padding = 'same', kernel_initializer = 'he_normal')(conv8)
        # conv8 = BatchNormalization()(conv8)
        up9 = Conv2D(64, 2, activation = 'relu', kernel_regularizer=regularizers.l2(weight_decay),padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
        merge9 = concatenate([conv1,up9], axis = 3)
        conv9 = Conv2D(64, 3, activation = 'relu', kernel_regularizer=regularizers.l2(weight_decay),padding = 'same', kernel_initializer = 'he_normal')(merge9)
        # conv9 = BatchNormalization()(conv9)
        conv9 = Conv2D(64, 3, activation = 'relu', kernel_regularizer=regularizers.l2(weight_decay),padding = 'same', kernel_initializer = 'he_normal')(conv9)
        # conv9 = BatchNormalization()(conv9)
        conv9 = Conv2D(2, 3, activation = 'relu', kernel_regularizer=regularizers.l2(weight_decay),padding = 'same', kernel_initializer = 'he_normal')(conv9)
        conv10 = Conv2D(1, 1, activation = 'sigmoid')(conv9)

        model = Model(input = img_input, output = conv10)
        return model
    def normalize(self, X_train, X_test):
        # this function normalize inputs for zero mean and unit variance
        # it is used when training a model.

        mean = np.mean(X_train, axis=(0, 1, 2, 3))
        std = np.std(X_train, axis=(0, 1, 2, 3))
        X_train = (X_train - mean) / (std + 1e-7)
        X_test = (X_test - mean) / (std + 1e-7)
        print('std,mean',std,mean)
        return X_train, X_test

    # def _load_data(self):
    #     # self.train_generator = DatasetGenerator(trainset,self.batch_size)
    #     # self.valid_generator = DatasetGenerator(testset,self.batch_size)
    #     (x_train, y_train) = load_cats_vs_dogs(extend='train_nbfs.npz')
    #     self.x_train = x_train.astype('float32')
    #     self.y_train = y_train.astype('float32')
    #     (x_test, y_test) = load_cats_vs_dogs(extend='test_nbfs.npz')
    #     self.x_test = x_test.astype('float32')
    #     self.y_test = y_test.astype('float32')
    #     self.x_train, self.x_test = self.normalize(x_train, x_test)

mod = CIFARUnet(train=False)
model = mod.build_model()
#manage_the_input
#feed the network
#read the input
data_path = 'T1Img/sub-01/T1w.nii.gz'
# data_dirs = os.listdir(data_path)
#get the output of the model for each slice
# print(data_dirs)
# filenames = 'A1...','A2....'
# for file in data_path:
#     print(file)
x = SliceGenerator(data_path)
# for i in range(x.shape[2]):

print(x[0].shape)
preds = model.predict(x[0])

#Use the Datset Generator
# model.evaluate

#Provide input to the model and collect outputs


