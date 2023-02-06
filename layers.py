#!/usr/bin/env python
import tensorflow as tf
from tensorflow.keras.layers import Conv3D, Activation, Add, BatchNormalization

@tf.keras.utils.register_keras_serializable()
class Residual(tf.keras.layers.Layer):
    def __init__(self, channels_in,kernel, strides=(1, 1, 1), batch_momentum=0.9, **kwargs):
        super(Residual, self).__init__(**kwargs)
        self.channels_in = channels_in
        self.kernel = kernel
        self.momentum = batch_momentum
        self.strides = strides

    def call(self, x):
        # the residual block using Keras functional API
        first_layer =   Activation("linear", trainable=False)(x)
        x =             Conv3D( self.channels_in,
                                self.kernel,
                                strides=self.strides,
                                dilation_rate=2,
                                padding="same")(first_layer)
        x =             Batchnormalizaition(momentum=self.momentum)(x)
        x =             Activation("relu")(x)
        x =             Conv3D( self.channels_in,
                                self.kernel,
                                dilation_rate=2,
                                padding="same")(x)
        x =             Batchnormalizaition(momentum=self.momentum)(x)
        residual =      Conv3D( self.channels_in,
                                (1, 1, 1),
                                padding="same")(first_layer)
        residual =      Add()([x, residual])
        x =             Activation("relu")(residual)
        return x

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        cfg = super().get_config()
        cfg['filters'] = self.channels_in
        cfg['kernel'] = self.kernel
        return cfg

    def build(self, input_shape):
        self.w = self.add_weight(shape=(input_shape[-1], self.channels_in),
                                 initializer='random_normal',
                                 trainable=True, name='w')
        self.b = self.add_weight(shape=(self.channels_in,),
                                 initializer='random_normal',
                                 trainable=True, name='b')
  
    def call(self, inputs):
        return tf.matmul(inputs, self.w) + self.b

    @classmethod
    def from_config(cls, config):
        return cls(**config)
