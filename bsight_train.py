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

data_path = "./"
def load_cats_vs_dogs(cats_vs_dogs_path=data_path,extend = 'train_nbfs.npz'):
    npz_file = np.load(os.path.join(cats_vs_dogs_path, extend))
    x_train = npz_file['x_train']
    y_train = npz_file['y_train']

    return (x_train, y_train)



class CIFARUnet:
    def __init__(self, train=True, filename = 'live',maxepochs = 250):
        # self.alpha = alpha
        self.num_classes = 1
        self.weight_decay= 1e-6
        self.weight_decay_fc= 5e-4
        self.weight_decay_rc= 5e-4
        # self.noise = noise_frac
        self.batch_size = 16
        self._load_data()
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
            self.model.load_weights("history_checkpoints/{}".format(self.filename))

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

    def _load_data(self):
            # self.train_generator = DatasetGenerator(trainset,self.batch_size)
            # self.valid_generator = DatasetGenerator(testset,self.batch_size)
            (x_train, y_train) = load_cats_vs_dogs(extend='train_nbfs.npz')
            self.x_train = x_train.astype('float32')
            self.y_train = y_train.astype('float32')
            (x_test, y_test) = load_cats_vs_dogs(extend='test_nbfs.npz')
            self.x_test = x_test.astype('float32')
            self.y_test = y_test.astype('float32')
            self.x_train, self.x_test = self.normalize(x_train, x_test)

    def train(self,model):
        bs = self.batch_size
        maxepoches = self.maxepochs
        learning_rate = 1e-1

        lr_decay = 1e-6

        lr_drop = 25
        
        def dice_coefficient_loss(y_true, y_pred):
            return 1-dice_coefficient(y_true, y_pred,1)
        def dice_coefficient(y_true, y_pred, smooth=1.):
            y_true_f = K.flatten(y_true)
            y_pred_f = K.flatten(y_pred)
            intersection = K.sum(y_true_f * y_pred_f)
            return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
        def lr_scheduler(epoch):
            return learning_rate * (0.5 ** (epoch // lr_drop))

        #reduce_lr = keras.callbacks.LearningRateScheduler(lr_scheduler)
        # from keras.utils import multi_gpu_model
        # data augmentation
        # model = multi_gpu_model(model, gpus=None)
        # from keras.preprocessing.image import ImageDataGenerator,standardize,random_transform
            # input generator with standardization on
        datagenX = ImageDataGenerator(
            featurewise_center=False,  # set input mean to 0 over the dataset
            zca_whitening=False,  # apply ZCA whitening
            rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
            width_shift_range=0,  # randomly shift images horizontally (fraction of total width)
            height_shift_range=0,  # randomly shift images vertically (fraction of total height)
            horizontal_flip=False,  # randomly flip images
            vertical_flip=False,
            samplewise_center=False,  # set each sample mean to 0
            featurewise_std_normalization=False,  # divide inputs by std of the dataset
            samplewise_std_normalization=False,  # divide each input by its std
            )

        # output generator with standardization off
        datagenY = ImageDataGenerator(
            featurewise_center=False,  # set input mean to 0 over the dataset
            zca_whitening=False,  # apply ZCA whitening
            rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
            width_shift_range=0,  # randomly shift images horizontally (fraction of total width)
            height_shift_range=0,  # randomly shift images vertically (fraction of total height)
            horizontal_flip=False,  # randomly flip images
            vertical_flip=False,
            samplewise_center=False,  # set each sample mean to 0
            featurewise_std_normalization=False,  # divide inputs by std of the dataset
            samplewise_std_normalization=False,  # divide each input by its std
            )
        datagenX.fit(self.x_train,seed=1)
        datagenY.fit(self.y_train,seed=1)
        reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss',factor=0.5,patience=20,min_lr=0.00001,min_delta=0.0001,verbose=1)
        
        datagenX_v = ImageDataGenerator(
            featurewise_center=False,  # set input mean to 0 over the dataset
            zca_whitening=False,  # apply ZCA whitening
            rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
            width_shift_range=0,  # randomly shift images horizontally (fraction of total width)
            height_shift_range=0,  # randomly shift images vertically (fraction of total height)
            horizontal_flip=False,  # randomly flip images
            vertical_flip=False,
            samplewise_center=False,  # set each sample mean to 0
            featurewise_std_normalization=False,  # divide inputs by std of the dataset
            samplewise_std_normalization=False,  # divide each input by its std
            )

        # output generator with standardization off
        datagenY_v = ImageDataGenerator(
            featurewise_center=False,  # set input mean to 0 over the dataset
            zca_whitening=False,  # apply ZCA whitening
            rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
            width_shift_range=0,  # randomly shift images horizontally (fraction of total width)
            height_shift_range=0,  # randomly shift images vertically (fraction of total height)
            horizontal_flip=False,  # randomly flip images
            vertical_flip=False,
            samplewise_center=False,  # set each sample mean to 0
            featurewise_std_normalization=False,  # divide inputs by std of the dataset
            samplewise_std_normalization=False,  # divide each input by its std
            )
        datagenX_v.fit(self.x_test,seed=1)
        datagenY_v.fit(self.y_test,seed=1)
        ep = 1e-07
        # optimization details
        sgd = optimizers.SGD(lr=learning_rate, decay=lr_decay, momentum=0.9, nesterov=True)
        adam = optimizers.Adam(lr=2e-4,epsilon=ep)
        model.compile(loss=dice_coefficient_loss,
                      optimizer=adam, metrics=[dice_coefficient])
        x_generator = datagenX.flow(self.x_train,batch_size=bs,seed=0)
        y_generator = datagenY.flow(self.y_train,batch_size=bs,seed=0)
        xv_gen = datagenX_v.flow(self.x_test,batch_size=bs,seed=0)
        yv_gen = datagenY_v.flow(self.y_test,batch_size=bs,seed=0)
        historytemp = model.fit_generator(zip(x_generator,y_generator),
                                          steps_per_epoch=self.x_train.shape[0] // bs,
                                          epochs=maxepoches, callbacks=[reduce_lr],
                                        initial_epoch=0,validation_steps=10,
                                        validation_data=zip(xv_gen,yv_gen))

        model.save_weights("checkpoints/{}".format(self.filename))
        return model
    
    
# model = CIFARUnet(True,'abcd',100)




    # def build_model(self):
    #     # Build the network of vgg for 10 classes with massive dropout and weight decay as described in the paper.
    #     acti = 'relu'
    #     # weight_decay = self.weight_decay
    #     weight_decay = self.weight_decay
    #     weight_decay_fc = self.weight_decay_fc
    #     weight_decay_rc = self.weight_decay_rc
    #     basic_dropout_rate = 0.3
    #     n_classes = 1
    #     # elif IMAGE_ORDERING == 'channels_last':
    #     MERGE_AXIS = -1
    #     IMAGE_ORDERING = 'channels_last'
    #     img_input = Input(shape=(256,256,1))
    #     conv1 = Conv2D(32, (3, 3), data_format=IMAGE_ORDERING,
    #                activation='relu', padding='same')(img_input)
    #     conv1 = BatchNormalization()(conv1)
    #     conv1 = Dropout(0.2)(conv1)
    #     conv1 = Conv2D(32, (3, 3), data_format=IMAGE_ORDERING,
    #                   activation='relu', padding='same')(conv1)
    #     conv1 = BatchNormalization()(conv1)
    #     pool1 = MaxPooling2D((2, 2), data_format=IMAGE_ORDERING)(conv1)

    #     conv2 = Conv2D(64, (3, 3), data_format=IMAGE_ORDERING,
    #                   activation='relu', padding='same')(pool1)
    #     conv2 = BatchNormalization()(conv2)
    #     conv2 = Dropout(0.2)(conv2)
    #     conv2 = Conv2D(64, (3, 3), data_format=IMAGE_ORDERING,
    #                   activation='relu', padding='same')(conv2)
    #     conv2 = BatchNormalization()(conv2)
    #     pool2 = MaxPooling2D((2, 2), data_format=IMAGE_ORDERING)(conv2)

    #     conv3 = Conv2D(128, (3, 3), data_format=IMAGE_ORDERING,
    #                   activation='relu', padding='same')(pool2)
    #     conv3 = BatchNormalization()(conv3)
    #     conv3 = Dropout(0.2)(conv3)
    #     conv3 = Conv2D(128, (3, 3), data_format=IMAGE_ORDERING,
    #                   activation='relu', padding='same')(conv3)
    #     conv3 = BatchNormalization()(conv3)

    #     up1 = concatenate([UpSampling2D((2, 2), data_format=IMAGE_ORDERING)(
    #         conv3), conv2], axis=MERGE_AXIS)
    #     conv4 = Conv2D(64, (3, 3), data_format=IMAGE_ORDERING,
    #                   activation='relu', padding='same')(up1)
    #     conv4 = BatchNormalization()(conv4)
    #     # conv4 = Dropout(0.2)(conv4)
    #     conv4 = Dropout(0.2)(conv4)
    #     conv4 = Conv2D(64, (3, 3), data_format=IMAGE_ORDERING,
    #                   activation='relu', padding='same')(conv4)
    #     conv4 = BatchNormalization()(conv4)
    #     up2 = concatenate([UpSampling2D((2, 2), data_format=IMAGE_ORDERING)(
    #         conv4), conv1], axis=MERGE_AXIS)
    #     conv4 = BatchNormalization()(up2)
    #     conv5 = Conv2D(32, (3, 3), data_format=IMAGE_ORDERING,
    #                   activation='relu', padding='same')(up2)
    #     conv5 = Dropout(0.2)(conv5)
    #     conv5 = Conv2D(32, (3, 3), data_format=IMAGE_ORDERING,
    #                   activation='relu', padding='same' , name="seg_feats")(conv5)

    #     o = Conv2D(n_classes, (1, 1), data_format=IMAGE_ORDERING,
    #               padding='same')(conv5)

    #     # model = get_segmentation_model(img_input, o)
    #     model = Model(inputs=[img_input], outputs=[o])
    #     model.model_name = "unet_mini"
    #     return model