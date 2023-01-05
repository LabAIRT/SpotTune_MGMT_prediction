#!/usr/bin/env python
import tensorflow as tf
print("TensorFlow version:", tf.__version__)
from tensorflow.keras.layers import Dense, Convolution1D, LSTM, Flatten
from tensorflow.keras.layers import BatchNormalization, Dropout, Concatenate
from tensorflow.keras.models import Model



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
        ed = Dense(100, activation='relu', kernel_initializer='lecun_uniform', name='comb_dense0')(ed)
        if batchnorm:
            ed = BatchNormalization(momentum=batchmomentum, name='comb_dense_batchnorm0')(ed)
        ed = Dropout(dropout_rate, name='df_dense_dropout0')(ed)
        ed = Dense(50, activation='relu', kernel_initializer='lecun_uniform', name='comb_dense1')(ed)
        if batchnorm:
            ed = BatchNormalization(momentum=batchmomentum, name='comb_dense_batchnorm1')(ed)
        ed = Dropout(dropout_rate, name='df_dense_dropout1')(ed)
        ed = Dense(50, activation='relu', kernel_initializer='lecun_uniform', name='comb_dense2')(ed)
        if batchnorm:
            ed = BatchNormalization(momentum=batchmomentum, name='comb_dense_batchnorm2')(ed)
        ed = Dropout(dropout_rate, name='df_dense_dropout2')(ed)
        ed = Dense(25, activation='relu', kernel_initializer='lecun_uniform', name='comb_dense3')(ed)
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
