#!/usr/bin/env python
import tensorflow as tf
from tensorflow.keras.layers import Dense, Convolution1D, Conv3D, Flatten, Normalization, Rescaling, LSTM, TimeDistributed
from tensorflow.keras.layers import BatchNormalization, Dropout, Concatenate, MaxPooling3D, GlobalAveragePooling3D, Reshape, Activation, LeakyReLU, ZeroPadding3D
from tensorflow.keras.models import Model 

from gbm_project.layers import Residual, Bottleneck


def train_model_comb(Inputs, dropout_rate=0.1, conv_active=True, rec_active=True, dense_active=True, batchnorm=True, batchmomentum=0.6):
    """
    Inputs: data inputs
    dropout_rate: Dropout rate
    conv_active: activate CNN layers
    dens_active: activate Dense layers
    batchnorm: do BatchNormalization
    batchmomentum: momentum to give to BatchNormalization
    """

    ed = BatchNormalization(momentum=batchmomentum, name='ed_input_batchnorm')(Inputs)

    # put CNN layers into active statement so they can be turned on/off
    if conv_active:
        ed = Convolution1D(64, 1, padding='same', kernel_initializer='lecun_uniform', activation='relu', name='ed_conv0')(ed)
        if batchnorm:
            ed = BatchNormalization(momentum=batchmomentum, name='ed_batchnorm0')(ed)
        ed = Dropout(dropout_rate, name='ed_dropout0')(ed)
        ed = Convolution1D(32, 1, padding='same', kernel_initializer='lecun_uniform', activation='relu', name='ed_conv1')(ed)
        if batchnorm:
            ed = BatchNormalization(momentum=batchmomentum, name='ed_batchnorm1')(ed)
        ed = Dropout(dropout_rate, name='ed_dropout1')(ed)
        ed = Convolution1D(32, 1, padding='same', kernel_initializer='lecun_uniform', activation='relu', name='ed_conv2')(ed)
        if batchnorm:
            ed = BatchNormalization(momentum=batchmomentum, name='ed_batchnorm2')(ed)
        ed = Dropout(dropout_rate, name='ed_dropout2')(ed)
        ed = Convolution1D(8, 1, padding='same', kernel_initializer='lecun_uniform', activation='relu', name='ed_conv3')(ed)

    else:
        ed = Convolution1D(1, 1, kernel_initializer='zeros', trainable=False, name='ed_conv_off')(ed)

    # Now the Recurrent LSTM layers
    if rec_active:
        ed = LSTM(150, go_backwards=True, implementation=2, name='ed_lstm')(ed)
        if batchnorm:
            ed = BatchNormalization(momentum=batchmomentum, name='ed_lstm_batchnorm')(ed)
        ed = Dropout(dropout_rate, name='ed_lstm_dropout')(ed)
    else:
        ed = Flatten()(ed)

    # Now Dense Layer(s)
    if dense_active:
        #ed = Dense(100, activation='relu', kernel_initializer='lecun_uniform', name='comb_dense0')(ed)
        #if batchnorm:
        #    ed = BatchNormalization(momentum=batchmomentum, name='comb_dense_batchnorm0')(ed)
        #ed = Dropout(dropout_rate, name='df_dense_dropout0')(ed)
        #ed = Dense(32, activation='relu', kernel_initializer='lecun_uniform', name='comb_dense1')(ed)
        #if batchnorm:
        #    ed = BatchNormalization(momentum=batchmomentum, name='comb_dense_batchnorm1')(ed)
        #ed = Dropout(dropout_rate, name='df_dense_dropout1')(ed)
        ed = Dense(32, activation='relu', kernel_initializer='lecun_uniform', name='comb_dense2')(ed)
        if batchnorm:
            ed = BatchNormalization(momentum=batchmomentum, name='comb_dense_batchnorm2')(ed)
        ed = Dropout(dropout_rate, name='df_dense_dropout2')(ed)
        ed = Dense(16, activation='relu', kernel_initializer='lecun_uniform', name='comb_dense3')(ed)
        if batchnorm:
            ed = BatchNormalization(momentum=batchmomentum, name='comb_dense_batchnorm3')(ed)
        ed = Dropout(dropout_rate, name='df_dense_dropout3')(ed)
    else:
        ed = Dense(100, activation='relu', kernel_initializer='lecun_uniform', name='comb_dense_conv')(ed)

    # output layer
    pred = Dense(1, activation='sigmoid', kernel_initializer='lecun_uniform', name='ID_pred')(ed)

    model = Model(Inputs, pred)
    return model



def train_model(Inputs, dropout_rate=0.1, conv_active=True, rec_active=True, dense_active=True, batchnorm=True, batchmomentum=0.6):
    """
    Inputs: data inputs
    dropout_rate: Dropout rate
    conv_active: activate CNN layers
    dens_active: activate Dense layers
    batchnorm: do BatchNormalization
    batchmomentum: momentum to give to BatchNormalization
    """

    ed = Inputs[:, 0]
    et = Inputs[:, 1]
    nc = Inputs[:, 2]

    ed = BatchNormalization(momentum=batchmomentum, name='ed_input_batchnorm')(ed)
    et = BatchNormalization(momentum=batchmomentum, name='et_input_batchnorm')(et)
    nc = BatchNormalization(momentum=batchmomentum, name='nc_input_batchnorm')(nc)

    # put CNN layers into active statement so they can be turned on/off
    if conv_active:
        # Edema/Invasion section
        ed = Convolution1D(64, 1, kernel_initializer='lecun_uniform', activation='relu', name='ed_conv0')(ed)
        if batchnorm:
            ed = BatchNormalization(momentum=batchmomentum, name='ed_batchnorm0')(ed)
        ed = Dropout(dropout_rate, name='ed_dropout0')(ed)
        ed = Convolution1D(32, 1, kernel_initializer='lecun_uniform', activation='relu', name='ed_conv1')(ed)
        if batchnorm:
            ed = BatchNormalization(momentum=batchmomentum, name='ed_batchnorm1')(ed)
        ed = Dropout(dropout_rate, name='ed_dropout1')(ed)
        ed = Convolution1D(32, 1, kernel_initializer='lecun_uniform', activation='relu', name='ed_conv2')(ed)
        if batchnorm:
            ed = BatchNormalization(momentum=batchmomentum, name='ed_batchnorm2')(ed)
        ed = Dropout(dropout_rate, name='ed_dropout2')(ed)
        ed = Convolution1D(8, 1, kernel_initializer='lecun_uniform', activation='relu', name='ed_conv3')(ed)

        # Enhancing Tumor section
        et = Convolution1D(32, 1, kernel_initializer='lecun_uniform', activation='relu', name='et_conv0')(et)
        if batchnorm:
            et = BatchNormalization(momentum=batchmomentum, name='et_batchnorm0')(et)
        et = Dropout(dropout_rate, name='et_dropout0')(et)
        et = Convolution1D(32, 1, kernel_initializer='lecun_uniform', activation='relu', name='et_conv1')(et)
        if batchnorm:
            et = BatchNormalization(momentum=batchmomentum, name='et_batchnorm1')(et)
        et = Dropout(dropout_rate, name='et_dropout1')(et)
        et = Convolution1D(32, 1, kernel_initializer='lecun_uniform', activation='relu', name='et_conv2')(et)
        if batchnorm:
            et = BatchNormalization(momentum=batchmomentum, name='et_batchnorm2')(et)
        et = Dropout(dropout_rate, name='et_dropout2')(et)
        et = Convolution1D(8, 1, kernel_initializer='lecun_uniform', activation='relu', name='et_conv3')(et)

        # Necrotic Core Section
        nc = Convolution1D(32, 1, kernel_initializer='lecun_uniform', activation='relu', name='nc_conv0')(nc)
        if batchnorm:
            nc = BatchNormalization(momentum=batchmomentum, name='nc_batchnorm0')(nc)
        nc = Dropout(dropout_rate, name='nc_dropout0')(nc)
        nc = Convolution1D(32, 1, kernel_initializer='lecun_uniform', activation='relu', name='nc_conv1')(nc)
        if batchnorm:
            nc = BatchNormalization(momentum=batchmomentum, name='nc_batchnorm1')(nc)
        nc = Dropout(dropout_rate, name='nc_dropout1')(nc)
        nc = Convolution1D(32, 1, kernel_initializer='lecun_uniform', activation='relu', name='nc_conv2')(nc)
        if batchnorm:
            nc = BatchNormalization(momentum=batchmomentum, name='nc_batchnorm2')(nc)
        nc = Dropout(dropout_rate, name='nc_dropout2')(nc)
        nc = Convolution1D(8, 1, kernel_initializer='lecun_uniform', activation='relu', name='nc_conv3')(nc)
    else:
        ed = Convolution1D(1, 1, kernel_initializer='zeros', trainable=False, name='ed_conv_off')(ed)
        et = Convolution1D(1, 1, kernel_initializer='zeros', trainable=False, name='et_conv_off')(et)
        nc = Convolution1D(1, 1, kernel_initializer='zeros', trainable=False, name='nc_conv_off')(nc)

    # Now the Recurrent LSTM layers
    if rec_active:
        ed = LSTM(150, go_backwards=True, implementation=2, name='ed_lstm')(ed)
        if batchnorm:
            ed = BatchNormalization(momentum=batchmomentum, name='ed_lstm_batchnorm')(ed)
        ed = Dropout(dropout_rate, name='ed_lstm_dropout')(ed)
        et = LSTM(150, go_backwards=True, implementation=2, name='et_lstm')(et)
        if batchnorm:
            et = BatchNormalization(momentum=batchmomentum, name='et_lstm_batchnorm')(et)
        et = Dropout(dropout_rate, name='et_lstm_dropout')(et)
        nc = LSTM(100, go_backwards=True, implementation=2, name='nc_lstm')(nc)
        if batchnorm:
            nc = BatchNormalization(momentum=batchmomentum, name='nc_lstm_batchnorm')(nc)
        nc = Dropout(dropout_rate, name='nc_lstm_dropout')(nc)
    else:
        ed = Flatten()(ed)
        et = Flatten()(et)
        nc = Flatten()(nc)
        #ed = LSTM(1, trainable=False, name='ed_lstm_off')(ed)
        #et = LSTM(1, trainable=False, name='et_lstm_off')(et)
        #nc = LSTM(1, trainable=False, name='nc_lstm_off')(nc)

    # Now combine segments to feed into a dense layer
    x_comb = Concatenate()([ed,et,nc])
    # Now Dense Layer(s)
    if dense_active:
        x_comb = Dense(200, activation='relu', kernel_initializer='lecun_uniform', name='comb_dense0')(x_comb)
        if batchnorm:
            x_comb = BatchNormalization(momentum=batchmomentum, name='comb_dense_batchnorm0')(x_comb)
        x_comb = Dropout(dropout_rate, name='df_dense_dropout0')(x_comb)
        x_comb = Dense(100, activation='relu', kernel_initializer='lecun_uniform', name='comb_dense1')(x_comb)
        if batchnorm:
            x_comb = BatchNormalization(momentum=batchmomentum, name='comb_dense_batchnorm1')(x_comb)
        x_comb = Dropout(dropout_rate, name='df_dense_dropout1')(x_comb)
        x_comb = Dense(50, activation='relu', kernel_initializer='lecun_uniform', name='comb_dense2')(x_comb)
        if batchnorm:
            x_comb = BatchNormalization(momentum=batchmomentum, name='comb_dense_batchnorm2')(x_comb)
        x_comb = Dropout(dropout_rate, name='df_dense_dropout2')(x_comb)
        x_comb = Dense(25, activation='relu', kernel_initializer='lecun_uniform', name='comb_dense3')(x_comb)
        if batchnorm:
            x_comb = BatchNormalization(momentum=batchmomentum, name='comb_dense_batchnorm3')(x_comb)
        x_comb = Dropout(dropout_rate, name='df_dense_dropout3')(x_comb)
    else:
        x_comb = Dense(100, activation='relu', kernel_initializer='lecun_uniform', name='comb_dense_conv')(x_comb)

    # output layer
    pred = Dense(1, activation='sigmoid', kernel_initializer='lecun_uniform', name='ID_pred')(x_comb)

    model = Model(Inputs, pred)
    return model


def train_model_sequential(Inputs, dropout_rate=0.1, batchmomentum=0.6):

    model = tf.keras.Sequential()

    model.add(Inputs)

    model.add(Conv3D(64, (5, 5, 5), activation='relu', name='ed_image_conv0'))
    model.add(BatchNormalization(momentum=batchmomentum, name='ed_image_batchnorm0'))
    #model.add(Dropout(dropout_rate, name='ed_image_dropout0'))

    model.add(Conv3D(64, (3, 3, 3), activation='relu', name='ed_image_conv1'))
    model.add(BatchNormalization(momentum=batchmomentum, name='ed_image_batchnorm1'))
    #model.add(Dropout(dropout_rate, name='ed_image_dropout1'))

    model.add(Conv3D(64, (3, 3, 3), activation='relu', name='ed_image_conv2'))
    model.add(BatchNormalization(momentum=batchmomentum, name='ed_image_batchnorm2'))
    #model.add(Dropout(dropout_rate, name='ed_image_dropout2'))

    model.add(Conv3D(64, (3, 3, 3), activation='relu', name='ed_image_conv3'))
    model.add(BatchNormalization(momentum=batchmomentum, name='ed_image_batchnorm3'))
    #model.add(Dropout(dropout_rate, name='ed_image_dropout3'))

    model.add(Conv3D(64, (3, 3, 3), activation='relu', name='ed_image_conv4'))
    model.add(BatchNormalization(momentum=batchmomentum, name='ed_image_batchnorm4'))
    #model.add(Dropout(dropout_rate, name='ed_image_dropout4'))

    model.add(MaxPooling3D((2, 2, 2), strides=(2, 2, 2), name='ed_mp0'))

    model.add(Conv3D(64, (3, 3, 3), activation='relu', name='ed_image_conv5'))
    model.add(BatchNormalization(momentum=batchmomentum, name='ed_image_batchnorm5'))
    #model.add(Dropout(dropout_rate, name='ed_image_dropout5'))

    model.add(Conv3D(64, (2, 2, 2), activation='relu', name='ed_image_conv6'))
    model.add(BatchNormalization(momentum=batchmomentum, name='ed_image_batchnorm6'))
    #model.add(Dropout(dropout_rate, name='ed_image_dropout6'))

    model.add(Conv3D(64, (3, 3, 3), activation='relu', name='ed_image_conv7'))
    model.add(BatchNormalization(momentum=batchmomentum, name='ed_image_batchnorm7'))
    #model.add(Dropout(dropout_rate, name='ed_image_dropout7'))

    model.add(MaxPooling3D((3, 3, 3), strides=(2, 2, 2), name='ed_mp1'))

    model.add(Conv3D(64, (3, 3, 3), padding='same', activation='relu', name='ed_image_conv8'))
    model.add(BatchNormalization(momentum=batchmomentum, name='ed_image_batchnorm8'))
    #model.add(Dropout(dropout_rate, name='ed_image_dropout8'))

    model.add(Conv3D(64, (3, 3, 1), padding='same', activation='relu', name='ed_image_conv9'))
    model.add(BatchNormalization(momentum=batchmomentum, name='ed_image_batchnorm9'))
    #model.add(Dropout(dropout_rate, name='ed_image_dropout9'))

    model.add(Conv3D(64, (3, 3, 1), padding='same', activation='relu', name='ed_image_conv10'))
    model.add(BatchNormalization(momentum=batchmomentum, name='ed_image_batchnorm10'))
    #model.add(Dropout(dropout_rate, name='ed_image_dropout10'))

    model.add(Conv3D(32, (3, 3, 1), padding='same', activation='relu', name='ed_image_conv11'))
    model.add(BatchNormalization(momentum=batchmomentum, name='ed_image_batchnorm11'))
    #model.add(Dropout(dropout_rate, name='ed_image_dropout11'))

    model.add(Flatten())

    model.add(Dense(64, activation='relu', name='comb_dense0'))
    model.add(BatchNormalization(momentum=batchmomentum, name='comb_dense_batchnorm0'))
    #model.add(Dropout(dropout_rate, name='df_dense_dropout0'))

    model.add(Dense(32, activation='relu', name='comb_dense3'))
    model.add(BatchNormalization(momentum=batchmomentum, name='comb_dense_batchnorm3'))
    model.add(Dropout(dropout_rate, name='df_dense_dropout3'))

    # output layer
    model.add(Dense(1, activation='sigmoid', name='ID_pred'))

    return model


def train_model_resnet10(Inputs, dropout_rate=0.1, batchmomentum=0.6, l2_reg=0.01, dilation=2):

    model = tf.keras.Sequential()

    model.add(Inputs)

    model.add(Conv3D(64, (7, 7, 7), strides=(2, 2, 2), activation='relu', activity_regularizer=tf.keras.regularizers.L2(l2_reg)))
    model.add(BatchNormalization(momentum=batchmomentum))

    model.add(MaxPooling3D((3, 3, 3), strides=(2, 2, 2)))

    model.add(Residual(32, (3, 3, 3), batch_momentum=batchmomentum, l2_reg=l2_reg, dilation=dilation))
    model.add(BatchNormalization(momentum=batchmomentum))
    model.add(Residual(64, (3, 3, 3), strides=2, batch_momentum=batchmomentum, l2_reg=l2_reg, dilation=dilation))
    model.add(BatchNormalization(momentum=batchmomentum))

    model.add(Residual(128, (3, 3, 3), strides=2, batch_momentum=batchmomentum, l2_reg=l2_reg, dilation=dilation))
    model.add(BatchNormalization(momentum=batchmomentum))

    model.add(Residual(256, (3, 3, 3), strides=2, batch_momentum=batchmomentum, l2_reg=l2_reg, dilation=dilation))
    model.add(BatchNormalization(momentum=batchmomentum))

    model.add(GlobalAveragePooling3D())

    model.add(Dense(100, activation='relu', name='comb_dense0', activity_regularizer=tf.keras.regularizers.L2(l2_reg)))
    model.add(BatchNormalization(momentum=batchmomentum, name='comb_dense_batchnorm0'))
    model.add(Dropout(dropout_rate))

    # output layer
    model.add(Dense(1, activation='sigmoid', name='ID_pred'))

    return model




def train_model_resnet(Inputs, dropout_rate=0.1, batchmomentum=0.6, l2_reg=0.05, dilation=1):

    model = tf.keras.Sequential()

    model.add(Inputs)

    model.add(TimeDistributed(Conv3D(64, (1, 7, 7), strides=(1, 2, 2), activation='relu', activity_regularizer=tf.keras.regularizers.L2(l2_reg))))
    model.add(BatchNormalization(momentum=batchmomentum))

    model.add(TimeDistributed(MaxPooling3D((1, 3, 3), strides=(1, 2, 2))))
    #model.add(Dropout(dropout_rate))

    model.add(Residual(32, (3, 3, 3), batch_momentum=batchmomentum, l2_reg=l2_reg, dilation=dilation))
    model.add(Residual(32, (3, 3, 3), batch_momentum=batchmomentum, l2_reg=l2_reg, dilation=dilation))

    model.add(Residual(64, (3, 3, 3), strides=(1,2,2), batch_momentum=batchmomentum, l2_reg=l2_reg, dilation=1))
    model.add(Residual(64, (3, 3, 3), batch_momentum=batchmomentum, l2_reg=l2_reg, dilation=dilation))

    model.add(Residual(128, (3, 3, 3), strides=(1,2,2), batch_momentum=batchmomentum, l2_reg=l2_reg, dilation=1))
    model.add(Residual(128, (3, 3, 3), batch_momentum=batchmomentum, l2_reg=l2_reg, dilation=dilation))

    model.add(Residual(256, (3, 3, 3), strides=(2,2,2), batch_momentum=batchmomentum, l2_reg=l2_reg, dilation=1))
    model.add(Residual(256, (3, 3, 3), batch_momentum=batchmomentum, l2_reg=l2_reg, dilation=dilation))

    model.add(Reshape((-1, 3, 3, 256)))
    model.add(GlobalAveragePooling3D())
    #model.add(Dropout(dropout_rate))


    model.add(Dense(128, activation='relu', activity_regularizer=tf.keras.regularizers.L2(l2_reg)))
    model.add(BatchNormalization(momentum=batchmomentum))
    model.add(Dropout(dropout_rate))
    #model.add(Dense(64, activation='relu', activity_regularizer=tf.keras.regularizers.L2(l2_reg)))
    #model.add(BatchNormalization(momentum=batchmomentum))
    #model.add(Dropout(dropout_rate))
    model.add(Dense(64, activation='relu', activity_regularizer=tf.keras.regularizers.L2(l2_reg)))
    model.add(BatchNormalization(momentum=batchmomentum))
    model.add(Dropout(dropout_rate))
    model.add(Dense(32, activation='relu', activity_regularizer=tf.keras.regularizers.L2(l2_reg)))
    model.add(BatchNormalization(momentum=batchmomentum))
    model.add(Dropout(dropout_rate))
    #model.add(Dense(32, activation='relu', activity_regularizer=tf.keras.regularizers.L2(l2_reg)))
    #model.add(BatchNormalization(momentum=batchmomentum))
    #model.add(Dropout(dropout_rate))
    #model.add(Dense(64, activation='relu', activity_regularizer=tf.keras.regularizers.L2(l2_reg)))
    #model.add(BatchNormalization(momentum=batchmomentum))
    #model.add(Dropout(dropout_rate))
    #model.add(Dense(64, activation='relu', activity_regularizer=tf.keras.regularizers.L2(l2_reg)))
    #model.add(BatchNormalization(momentum=batchmomentum))
    #model.add(Dropout(dropout_rate))
    #model.add(Dense(16, activation='relu', activity_regularizer=tf.keras.regularizers.L2(l2_reg)))
    #model.add(BatchNormalization(momentum=batchmomentum))
    #model.add(Dropout(dropout_rate))
    #model.add(Dense(8, activation='relu', activity_regularizer=tf.keras.regularizers.L2(l2_reg)))
    #model.add(BatchNormalization(momentum=batchmomentum))
    #model.add(Dropout(dropout_rate))
    #model.add(Dense(4, activation='relu', activity_regularizer=tf.keras.regularizers.L2(l2_reg)))
    #model.add(BatchNormalization(momentum=batchmomentum))
    #model.add(Dropout(dropout_rate))
    #model.add(Dense(2, activation='relu', activity_regularizer=tf.keras.regularizers.L2(l2_reg)))
    #model.add(BatchNormalization(momentum=batchmomentum))
    #model.add(Dropout(dropout_rate))



    # output layer
    model.add(Dense(1, activation='sigmoid', name='ID_pred'))

    return model


def train_model_resnet34(Inputs, dropout_rate=0.1, batchmomentum=0.6):

    model = tf.keras.Sequential()

    model.add(Inputs)

    model.add(Conv3D(64, (7, 7, 7), strides=(2, 2, 2), activation='relu', name='ed_image_conv0'))
    model.add(BatchNormalization(momentum=batchmomentum, name='ed_image_batchnorm0'))

    model.add(MaxPooling3D((3, 3, 3), strides=(2, 2, 2), name='ed_mp0'))
    model.add(Dropout(dropout_rate))

    model.add(Residual(64, (3, 3, 3), batch_momentum=batchmomentum))
    model.add(Residual(64, (3, 3, 3), batch_momentum=batchmomentum))
    model.add(Residual(64, (3, 3, 3), batch_momentum=batchmomentum))

    model.add(Residual(128, (3, 3, 3), strides=2, batch_momentum=batchmomentum))
    model.add(Residual(128, (3, 3, 3), batch_momentum=batchmomentum))
    model.add(Residual(128, (3, 3, 3), batch_momentum=batchmomentum))
    model.add(Residual(128, (3, 3, 3), batch_momentum=batchmomentum))

    model.add(Residual(256, (3, 3, 3), strides=2, batch_momentum=batchmomentum))
    model.add(Residual(256, (3, 3, 3), batch_momentum=batchmomentum))
    model.add(Residual(256, (3, 3, 3), batch_momentum=batchmomentum))
    model.add(Residual(256, (3, 3, 3), batch_momentum=batchmomentum))
    model.add(Residual(256, (3, 3, 3), batch_momentum=batchmomentum))
    model.add(Residual(256, (3, 3, 3), batch_momentum=batchmomentum))

    model.add(Residual(512, (3, 3, 3), strides=2, batch_momentum=batchmomentum))
    model.add(Residual(512, (3, 3, 3), batch_momentum=batchmomentum))
    model.add(Residual(512, (3, 3, 3), batch_momentum=batchmomentum))

    model.add(GlobalAveragePooling3D())
    model.add(Dropout(dropout_rate))

    model.add(Dense(100, activation='relu', name='comb_dense0'))
    model.add(BatchNormalization(momentum=batchmomentum, name='comb_dense_batchnorm0'))
    model.add(Dropout(dropout_rate, name='df_dense_dropout0'))

    #model.add(Dense(50, activation='relu', name='comb_dense3'))
    #model.add(BatchNormalization(momentum=batchmomentum, name='comb_dense_batchnorm3'))
    #model.add(Dropout(dropout_rate, name='df_dense_dropout3'))

    # output layer
    model.add(Dense(1, activation='sigmoid', name='ID_pred'))

    return model


def train_model_resnet50(Inputs, dropout_rate=0.1, batchmomentum=0.6, l2_reg=0.05, dilation=1):
    model = tf.keras.Sequential()

    model.add(Inputs)
    model.add(ZeroPadding3D(((3, 3), (3, 3), (3, 3))))
    model.add(Conv3D(64, (7, 7, 7), strides=(2, 2, 2), name='conv1_conv'))
    model.add(ZeroPadding3D(((1, 1), (1, 1), (1, 1))))
    model.add(MaxPooling3D((3, 3, 3), strides=(2, 2, 2)))

    #Conv 2
    model.add(BatchNormalization(momentum=batchmomentum), name='conv2_block1_preact_bn')
    model.add(Activation('relu'), name='conv2_block1_preact_relu')

    model.add(Bottleneck(256, 64, (3, 3, 3), preact=False, name='conv2_block1'))
    model.add(Bottleneck(256, 64, (3, 3, 3), preact=True), name='conv2_block2')
    model.add(Bottleneck(256, 64, (3, 3, 3), strides=2, preact=True, pool=True, name='conv2_block3'))

    #Conv 3
    model.add(BatchNormalization(momentum=batchmomentum), name='conv3_block1_preact_bn')
    model.add(Activation('relu'), name='conv3_block1_preact_relu')

    model.add(Bottleneck(512, 128, (3, 3, 3), preact=False), name='conv3_block1')
    model.add(Bottleneck(512, 128, (3, 3, 3), preact=True), name='conv3_block2') 
    model.add(Bottleneck(512, 128, (3, 3, 3), preact=True), name='conv3_block3') 
    model.add(Bottleneck(512, 128, (3, 3, 3), strides=2, preact=True, pool=True, name='conv3_block4'))

    #Conv 4
    model.add(BatchNormalization(momentum=batchmomentum), name='conv4_block1_preact_bn')
    model.add(Activation('relu'), name='conv4_block1_preact_relu')

    model.add(Bottleneck(1024, 256, (3, 3, 3), preact=False), name='conv4_block1')
    model.add(Bottleneck(1024, 256, (3, 3, 3), preact=True), name='conv4_block2') 
    model.add(Bottleneck(1024, 256, (3, 3, 3), preact=True), name='conv4_block3') 
    model.add(Bottleneck(1024, 256, (3, 3, 3), preact=True), name='conv4_block4') 
    model.add(Bottleneck(1024, 256, (3, 3, 3), preact=True), name='conv4_block5') 
    model.add(Bottleneck(1024, 256, (3, 3, 3), strides=2, preact=True, pool=True, name='conv4_block6'))

    #Conv 5
    model.add(BatchNormalization(momentum=batchmomentum), name='conv5_block1_preact_bn')
    model.add(Activation('relu'), name='conv5_block1_preact_relu')

    model.add(Bottleneck(2048, 512, (3, 3, 3), preact=False), name='conv5_block1')
    model.add(Bottleneck(2048, 512, (3, 3, 3), preact=True), name='conv5_block2') 
    model.add(Bottleneck(2048, 512, (3, 3, 3), preact=True), name='conv5_block3')

    model.add(BatchNormalization(momentum=batchmomentum), name='post_bn')
    model.add(Activation('relu'), name='post_relu')

    model.add(GlobalAveragePooling3D(), name='avg_pool')

    model.add(Dense(1, activation='sigmoid')) 

    return model


def train_model_resnet_timedistributed(Inputs, dropout_rate=0.1, batchmomentum=0.6, l2_reg=0.05, dilation=1):

    model = tf.keras.Sequential()

    model.add(Inputs)

    model.add(TimeDistributed(Conv3D(64, (5, 7, 7), strides=(1, 2, 2), activation='relu', activity_regularizer=tf.keras.regularizers.L2(l2_reg))))
    model.add(TimeDistributed(BatchNormalization(momentum=batchmomentum)))

    model.add(TimeDistributed(MaxPooling3D((1, 3, 3), strides=(1, 2, 2))))
    model.add(TimeDistributed(Dropout(dropout_rate)))

    model.add(TimeDistributed(Residual(64, (3, 3, 3), batch_momentum=batchmomentum, l2_reg=l2_reg, dilation=dilation)))
    model.add(TimeDistributed(BatchNormalization(momentum=batchmomentum)))
    model.add(TimeDistributed(Residual(64, (3, 3, 3), batch_momentum=batchmomentum, l2_reg=l2_reg, dilation=dilation)))
    model.add(TimeDistributed(BatchNormalization(momentum=batchmomentum)))

    model.add(TimeDistributed(Residual(128, (3, 3, 3), strides=(1, 2, 2), batch_momentum=batchmomentum, l2_reg=l2_reg, dilation=1)))
    model.add(TimeDistributed(BatchNormalization(momentum=batchmomentum)))
    model.add(TimeDistributed(Residual(128, (3, 3, 3), batch_momentum=batchmomentum, l2_reg=l2_reg, dilation=dilation)))
    model.add(TimeDistributed(BatchNormalization(momentum=batchmomentum)))

    model.add(TimeDistributed(Residual(256, (3, 3, 3), strides=(1, 2, 2), batch_momentum=batchmomentum, l2_reg=l2_reg, dilation=1)))
    model.add(TimeDistributed(BatchNormalization(momentum=batchmomentum)))
    model.add(TimeDistributed(Residual(256, (3, 3, 3), batch_momentum=batchmomentum, l2_reg=l2_reg, dilation=dilation)))
    model.add(TimeDistributed(BatchNormalization(momentum=batchmomentum)))

    model.add(TimeDistributed(Residual(512, (3, 3, 3), strides=(1, 2, 2), batch_momentum=batchmomentum, l2_reg=l2_reg, dilation=1)))
    model.add(TimeDistributed(BatchNormalization(momentum=batchmomentum)))
    model.add(TimeDistributed(Residual(512, (3, 3, 3), batch_momentum=batchmomentum, l2_reg=l2_reg, dilation=dilation)))
    model.add(TimeDistributed(BatchNormalization(momentum=batchmomentum)))

    model.add(TimeDistributed(GlobalAveragePooling3D()))
    model.add(TimeDistributed(Dropout(dropout_rate)))


    model.add(Dense(128, activation='relu', activity_regularizer=tf.keras.regularizers.L2(l2_reg)))
    model.add(BatchNormalization(momentum=batchmomentum))
    model.add(Dropout(dropout_rate))
    model.add(Dense(64, activation='relu', activity_regularizer=tf.keras.regularizers.L2(l2_reg)))
    model.add(BatchNormalization(momentum=batchmomentum))
    model.add(Dropout(dropout_rate))
    model.add(Dense(32, activation='relu', activity_regularizer=tf.keras.regularizers.L2(l2_reg)))
    model.add(BatchNormalization(momentum=batchmomentum))
    model.add(Dropout(dropout_rate))


    # output layer
    model.add(Dense(1, activation='sigmoid', name='ID_pred'))

    return model
