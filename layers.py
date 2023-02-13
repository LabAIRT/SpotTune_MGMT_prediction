#!/usr/bin/env python
import tensorflow as tf
from tensorflow.keras.layers import Conv3D, Activation, Add, BatchNormalization

@tf.keras.utils.register_keras_serializable()
class Residual(tf.keras.layers.Layer):
    def __init__(self, channels_in,kernel, strides=(1, 1, 1), batch_momentum=0.9, l2_reg=0.01, dilation=1, **kwargs):
        super(Residual, self).__init__(**kwargs)
        self.channels_in = channels_in
        self.kernel = kernel
        self.momentum = batch_momentum
        self.strides = strides
        self.l2_reg = l2_reg
        self.dilation= dilation
        
    def build(self, input_shape):
        self.linact = Activation('linear', trainable=False)
        self.conv1 =  Conv3D( self.channels_in,
                          self.kernel,
                          strides=self.strides,
                          dilation_rate=self.dilation,
                          activity_regularizer=tf.keras.regularizers.L2(self.l2_reg),
                          padding='same')
        self.bn1 =    BatchNormalization(momentum=self.momentum)
        self.act1 =   Activation('relu') 
        self.conv2 =  Conv3D( self.channels_in,
                          self.kernel,
                          strides=1,
                          dilation_rate=1,
                          activity_regularizer=tf.keras.regularizers.L2(self.l2_reg),
                          padding='same')
        self.bn2 =    BatchNormalization(momentum=self.momentum)
        self.residual = Conv3D(self.channels_in,
                               (1, 1, 1),
                               strides=self.strides,
                               dilation_rate = self.dilation,
                               padding='same')
        self.add =    Add()
        self.last_act = Activation('relu')
        super().build(input_shape)


    def call(self, x):
        input_layer = self.linact(x)
        x = self.conv1(input_layer)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        residual = self.residual(input_layer)
        x = self.add([x, residual])
        x = self.last_act(x)

        return x

    def get_config(self):
        cfg = super().get_config()
        cfg['filters'] = self.channels_in
        cfg['kernel'] = self.kernel
        return cfg

    @classmethod
    def from_config(cls, config):
        return cls(**config)



@tf.keras.utils.register_keras_serializable()
class Bottleneck(tf.keras.layers.Layer):
    def __init__(self, channels_in, channels_bottle, kernel, strides=(1, 1, 1), batch_momentum=0.99, l2_reg=0, dilation=1, preact=True, pool=False, **kwargs):
        super(Residual, self).__init__(**kwargs)
        self.channels_in = channels_in
        self.channels_bottle = channels_bottle
        self.kernel = kernel
        self.momentum = batch_momentum
        self.strides = strides
        self.l2_reg = l2_reg
        self.dilation= dilation
        self.preact = preact
        self.pool = pool
       
    def build(self, input_shape):
        if self.preact:
            self.prebn =    BatchNormalization(momentum=self.momentum)
            self.preact = Activation('relu', trainable=True)
        self.conv1 =  Conv3D( self.channels_bottle,
                          1,
                          strides=1,
                          dilation_rate=1,
                          activity_regularizer=tf.keras.regularizers.L2(self.l2_reg),
                          padding='same')
        self.bn1 =    BatchNormalization(momentum=self.momentum)
        self.act1 =   Activation('relu') 
        self.conv2 =  Conv3D( self.channels_bottle,
                          self.kernel,
                          strides=self.strides,
                          dilation_rate=self.dilation,
                          activity_regularizer=tf.keras.regularizers.L2(self.l2_reg),
                          padding='same')
        self.bn2 =    BatchNormalization(momentum=self.momentum)
        self.act2 =   Activation('relu') 
        self.conv3 =  Conv3D( self.channels_in,
                          1,
                          strides=1,
                          dilation_rate=1,
                          activity_regularizer=tf.keras.regularizers.L2(self.l2_reg),
                          padding='same')
        if self.pool:
            self.residual = MaxPooling3D(1, strides=self.strides)
        else:
            self.residual = Conv3D(self.channels_in,
                                   (1, 1, 1),
                                   strides=self.strides,
                                   dilation_rate = self.dilation,
                                   padding='same')
        self.add =    Add()
        super().build(input_shape)


    def call(self, x):

        residual = self.residual(x)

        if self.preact:
            x = self.prebn(x)
            x = self.preact(x)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act2(x)
        x = self.conv3(x)

        x = self.add([x, residual])

        return x

    def get_config(self):
        cfg = super().get_config()
        cfg['filters'] = self.channels_in
        cfg['kernel'] = self.kernel
        return cfg

    @classmethod
    def from_config(cls, config):
        return cls(**config)
