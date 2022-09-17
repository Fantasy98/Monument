import tensorflow as tf 
from tensorflow import keras
import keras.backend as K 
from keras import layers
import matplotlib.pyplot as plt 
import os 
import numpy as np


def Generator_Encoder():
    input_layer = layers.Input(name="input",shape=(HEIGHT,WIDTH,CHANNEL))
    x = layers.Conv2D(32,(5,5),strides=(1,1),padding ="same",name= "conv_1",kernel_regularizer="l2")(input_layer)
    x = layers.LeakyReLU(name="leaky_1")(x)
    x = layers.MaxPooling2D()(x)


    x = layers.Conv2D(64,(3,3),strides=(2,2),padding ="same",name= "conv_2",kernel_regularizer="l2")(x)
    x = layers.BatchNormalization(name="norm_1")(x)
    x = layers.LeakyReLU(name="leaky_2")(x)

    x = layers.MaxPooling2D()(x)


    x = layers.Conv2D(128,(3,3),strides=(2,2),padding ="same",name= "conv_3",kernel_regularizer="l2")(x)
    x = layers.BatchNormalization(name="norm_2")(x)
    x = layers.LeakyReLU(name="leaky_3")(x)
    
    # x = layers.MaxPooling2D()(x)
    
    x = layers.Conv2D(240,(3,3),strides=(2,2),padding ="same",name= "conv_4",kernel_regularizer="l2")(x)
    x = layers.BatchNormalization(name="norm_3")(x)
    x = layers.LeakyReLU(name="leaky_4")(x)

    # x = layers.Conv2D(512,(3,3),strides=(2,2),padding ="same",name= "conv_5",kernel_regularizer="l2")(x)
    # x = layers.BatchNormalization(name="norm_4")(x)
    # x = layers.LeakyReLU(name="leaky_5")(x)

    # x = layers.Conv2D(1024,(3,3),strides=(2,2),padding ="same",name= "conv_6",kernel_regularizer="l2")(x)
    # x = layers.BatchNormalization(name="norm_5")(x)
    # x = layers.LeakyReLU(name="leaky_6")(x)

    x = layers.GlobalAveragePooling2D(name = "g_encoder_output")(x)

    return keras.Model(inputs =input_layer,outputs =x)


def Generator():
    g_e  = Generator_Encoder()
    input_layer = layers.Input(name="input",shape=(HEIGHT,WIDTH,CHANNEL))

    x = g_e(input_layer)
    height = np.int64(HEIGHT//30)
    width = np.int64(WIDTH//32)
    x = layers.Dense(height*width*240,name="dense")(x)
    x = layers.Reshape((height,width,240),name="de_reshape")(x)

    x = layers.Conv2DTranspose(240,(4,4),strides =(2,2),padding = "same",name="deconv_1",kernel_regularizer="l2")(x)
    x = layers.LeakyReLU(name="de_leaky_1")(x)

    # x = layers.Conv2DTranspose(256,(3,3),strides =(2,2),padding = "same",name="deconv_2",kernel_regularizer="l2")(x)
    # x = layers.LeakyReLU(name="de_leaky_2")(x)

    x = layers.Conv2DTranspose(128,(2,2),strides =(2,2),padding = "same",name="deconv_3",kernel_regularizer="l2")(x)
    x = layers.LeakyReLU(name="de_leaky_3")(x)
    

    x = layers.Conv2DTranspose(64,(2,2),strides =(2,2),padding = "same",name="deconv_4",kernel_regularizer="l2")(x)
    x = layers.LeakyReLU(name="de_leaky_4")(x)

    x = layers.Conv2DTranspose(32,(2,2),strides =(2,2),padding = "same",name="deconv_5",kernel_regularizer="l2")(x)
    x = layers.LeakyReLU(name="de_leaky_5")(x)
    
    x = layers.Conv2DTranspose(16,(2,2),strides =(2,2),padding = "same",name="deconv_6",kernel_regularizer="l2")(x)
    x = layers.LeakyReLU(name="de_leaky_6")(x)
    

    x = layers.Conv2DTranspose(CHANNEL,(1,1),strides= (1,1),padding ="same",name="decoder_deconv_output",kernel_regularizer = "l2",activation="tanh")(x)

    x = layers.Resizing(HEIGHT,WIDTH)(x)
    return keras.Model(inputs =input_layer,outputs=x)
    
def Encoder():
    input_layer = layers.Input(name="encoder_input",shape=(HEIGHT,WIDTH,CHANNEL))
    x = layers.Conv2D(32,(5,5),strides=(1,1),padding ="same",name= "encoder_conv_1",kernel_regularizer="l2")(input_layer)
    x = layers.LeakyReLU(name="encoder_leaky_1")(x)
    x = layers.MaxPooling2D()(x)


    x = layers.Conv2D(64,(3,3),strides=(2,2),padding ="same",name= "encoder_conv_2",kernel_regularizer="l2")(x)
    x = layers.BatchNormalization(name="encoder_norm_1")(x)
    x = layers.LeakyReLU(name="encoder_leaky_2")(x)

    x = layers.MaxPooling2D()(x)


    x = layers.Conv2D(128,(3,3),strides=(2,2),padding ="same",name= "encoder_conv_3",kernel_regularizer="l2")(x)
    x = layers.BatchNormalization(name="encoder_norm_2")(x)
    x = layers.LeakyReLU(name="encoder_leaky_3")(x)
    
    # x = layers.MaxPooling2D()(x)
    
    x = layers.Conv2D(240,(3,3),strides=(2,2),padding ="same",name= "encoder_conv_4",kernel_regularizer="l2")(x)
    x = layers.BatchNormalization(name="encoder_norm_3")(x)
    x = layers.LeakyReLU(name="encoder_leaky_4")(x)

    # x = layers.Conv2D(512,(3,3),strides=(2,2),padding ="same",name= "conv_5",kernel_regularizer="l2")(x)
    # x = layers.BatchNormalization(name="norm_4")(x)
    # x = layers.LeakyReLU(name="leaky_5")(x)

    # x = layers.Conv2D(1024,(3,3),strides=(2,2),padding ="same",name= "conv_6",kernel_regularizer="l2")(x)
    # x = layers.BatchNormalization(name="norm_5")(x)
    # x = layers.LeakyReLU(name="leaky_6")(x)

    x = layers.GlobalAveragePooling2D(name = "encoder_output")(x)

    return keras.Model(inputs =input_layer,outputs =x)


def Feature_Extractor():
    input_layer = layers.Input(name="extractor_input",shape=(HEIGHT,WIDTH,CHANNEL))
    x = layers.Conv2D(32,(5,5),strides=(1,1),padding ="same",name= "extractor_conv_1",kernel_regularizer="l2")(input_layer)
    x = layers.LeakyReLU(name="extractor_leaky_1")(x)
    x = layers.MaxPooling2D()(x)


    x = layers.Conv2D(64,(3,3),strides=(2,2),padding ="same",name= "extractor_conv_2",kernel_regularizer="l2")(x)
    x = layers.BatchNormalization(name="extractor_norm_1")(x)
    x = layers.LeakyReLU(name="extractor_leaky_2")(x)

    x = layers.MaxPooling2D()(x)


    x = layers.Conv2D(128,(3,3),strides=(2,2),padding ="same",name= "extractor_conv_3",kernel_regularizer="l2")(x)
    x = layers.BatchNormalization(name="extractor_norm_2")(x)
    x = layers.LeakyReLU(name="extractor_leaky_3")(x)
    
    # x = layers.MaxPooling2D()(x)
    
    x = layers.Conv2D(240,(3,3),strides=(2,2),padding ="same",name= "extractor_conv_4",kernel_regularizer="l2")(x)
    x = layers.BatchNormalization(name="extractor_norm_3")(x)
    x = layers.LeakyReLU(name="extractor_leaky_4")(x)

    # x = layers.Conv2D(512,(3,3),strides=(2,2),padding ="same",name= "conv_5",kernel_regularizer="l2")(x)
    # x = layers.BatchNormalization(name="norm_4")(x)
    # x = layers.LeakyReLU(name="leaky_5")(x)

    # x = layers.Conv2D(1024,(3,3),strides=(2,2),padding ="same",name= "conv_6",kernel_regularizer="l2")(x)
    # x = layers.BatchNormalization(name="norm_5")(x)
    # x = layers.LeakyReLU(name="leaky_6")(x)

    # x = layers.GlobalAveragePooling2D(name = "extractor_output")(x)

    return keras.Model(inputs =input_layer,outputs =x)


def Discrimator():
    feature_extractor = Feature_Extractor()

    input_layer= layers.Input(name="input",shape=(HEIGHT,WIDTH,CHANNEL))
    x = feature_extractor(input_layer)

    x = layers.GlobalAveragePooling2D(name="glb_avg")(x)
    x = layers.Dense(1,activation = "sigmoid",name="d_output")(x)
    return keras.Model(input_layer,x)


class AdvLoss(keras.layers.Layer):
    def __init__(self,**kwargs):
        super(AdvLoss,self).__init__(**kwargs)
    def call(self,x,mask=None):
        ori_feature = feature_extractor(x[0])
        gan_feature = feature_extractor(x[1])

        return K.mean(K.square(ori_feature-K.mean(gan_feature,axis=0) ) )

    def get_output_shape_for(self,input_shape):
        return (input_shape[0][0],3)


class CntLoss(keras.layers.Layer):
    def __init__(self,**kwargs) :
        super(CntLoss,self).__init__(**kwargs)
        
    def call(self,x,mask=None):
        ori = x[0]
        gan = x[1]

        return K.mean(K.abs(ori-gan))

    def get_output_shape_for(self,input_shape):
        return (input_shape[0][0],3)

class EncLoss(keras.layers.Layer):
    def __init__(self,**kwargs):
        super(EncLoss,self).__init__(**kwargs)
    
    def call(self,x,mask=None):
        ori = x[0]
        gan = x[1]
        
        return K.mean(K.square( g_e(ori)-encoder(gan) ))
    
    def get_output_shape_for(self,input_shape):
        return (input_shape[0][0],3)

    