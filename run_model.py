#!/usr/bin/env python

import os, json
from datetime import datetime
import tensorflow as tf
print("TensorFlow version:", tf.__version__)
from tensorflow.keras.models import Model
from models import *
from data_prep import *


def run_model(config_file, X_train, y_train, components):
    with open(config_file, 'r') as f:
        config = json.load(f)

    n_batch = config['n_batch']
    n_epochs = config['n_epochs']
    learning_rate = config['learning_rate']
    dropout = config['dropout']

    # set up the keras Input object, shape corresponds to the 3x144 array per patient in X_scaled
    # for separate tumor sections
    #Inputs = tf.keras.Input(shape=(3,components,1), batch_size=n_batch)
    #for combined tumor sections
    Inputs = tf.keras.Input(shape=(components,3), batch_size=n_batch)
    #Inputs = tf.keras.Input(shape=(components,1), batch_size=n_batch)

    model = train_model_comb(Inputs, dropout_rate=dropout, rec_active=False, dense_active=True, conv_active=True)
    #model = train_model(Inputs, dropout_rate=dropout, rec_active=False, dense_active=True, conv_active=True)

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss='binary_crossentropy',
                  metrics=['accuracy', 'AUC'])
    print(model.summary())
    logdir = os.path.join('logs', datetime.now().strftime("%Y%m%d-%H%M%S"))
    tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)

    history = model.fit(X_train, 
              y_train,
              batch_size=n_batch,
              epochs=n_epochs,
              validation_split=0.25,
              verbose=1,
              callbacks=[tensorboard_callback])

    return history, model, logdir


def run_image_model(config_file, X_train, y_train):
    with open(config_file, 'r') as f:
        config = json.load(f)

    n_batch = config['n_batch']
    n_epochs = config['n_epochs']
    learning_rate = config['learning_rate']
    dropout = config['dropout']

    Inputs = tf.keras.Input(shape=(3,155,240,240,1), batch_size=n_batch)

    model = train_model_image(Inputs, dropout_rate=dropout)

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss='binary_crossentropy',
                  metrics=['accuracy', 'AUC'])

    print(model.summary())
    logdir = os.path.join('logs', datetime.now().strftime("%Y%m%d-%H%M%S"))
    tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)

    history = model.fit(X_train, 
              y_train,
              batch_size=n_batch,
              epochs=n_epochs,
              validation_split=0.25,
              verbose=1,
              callbacks=[tensorboard_callback])

    return history, model, logdir


def run_image_radiomic_model(config_file, X_train, y_train, components):
    with open(config_file, 'r') as f:
        config = json.load(f)

    n_batch = config['n_batch']
    n_epochs = config['n_epochs']
    learning_rate = config['learning_rate']
    dropout = config['dropout']

    # set up the keras Input object, shape corresponds to the 3x144 array per patient in X_scaled
    # for separate tumor sections
    Inputs = tf.keras.Input(shape=(6,None,None), batch_size=n_batch)
    #for combined tumor sections

    model = train_model_comb(Inputs, dropout_rate=dropout, rec_active=False, dense_active=True, conv_active=True)

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss='binary_crossentropy',
                  metrics=['accuracy', 'AUC'])
    print(model.summary())
    logdir = os.path.join('logs', datetime.now().strftime("%Y%m%d-%H%M%S"))
    tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)

    history = model.fit(X_train, 
              y_train,
              batch_size=n_batch,
              epochs=n_epochs,
              verbose=1,
              callbacks=[tensorboard_callback])

    return history, model, logdir


if __name__ == '__main__':
    #%tensorboard --logdir='./logs'
    
    csv_dir = 'D:/work/data/upenn_GBM/csvs/radiomic_features_CaPTk/'

    t1_df, man_df, comb_df = retrieve_data(csv_dir)
    X_train, X_test, y_train, y_test = scale_and_split(t1_df)

    n_test = len(X_test)
    n_train = len(X_train)
    mod_test_batch = n_test - n_test%n_batch
    mod_train_batch = n_train - n_train%n_batch

    # adjusts datasets so that they are divisble by the number of batches
    X_train_adj = X_train[0:mod_train_batch]
    y_train_adj = y_train[0:mod_train_batch]
    X_test_adj = X_test[0:mod_test_batch]
    y_test_adj = y_test[0:mod_test_batch]

    config_file = 'config_initial_20221228.json'
    history, model = run_model(config_file, X_train_adj, y_train_adj)

    results_test = model.evaluate(X_test_adj, y_test_adj, batch_size=n_batch)
    results_train = model.evaluate(X_train_adj, y_train_adj, batch_size=n_batch)
    model.save(logdir)
    print('evaluate on test data')
    print('test loss, test acc, test auc:', results_test)
    print('evaluate on train data')
    print('train loss, train acc, train auc:', results_train)
    y_pred = model.predict(X_test_adj, batch_size=n_batch)
    y_pred_train = model.predict(X_train_adj, batch_size=n_batch)
    display(y_pred)
    display(y_pred_train)
