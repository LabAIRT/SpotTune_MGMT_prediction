#!/usr/bin/env python

import os, json
from datetime import datetime
import tensorflow as tf
print("TensorFlow version:", tf.__version__)
from tensorflow.keras.models import Model

from sklearn.utils.class_weight import compute_class_weight

from gbm_project.models import *
from gbm_project.data_prep import *
from gbm_project.data_generator import data_generator
from gbm_project.data_generator import DataGenerator

def run_model(config_file, X_train, y_train, X_val, y_val, components):
    with open(config_file, 'r') as f:
        config = json.load(f)

    n_batch = config['n_batch']
    n_epochs = config['n_epochs']
    learning_rate = config['learning_rate']
    dropout = config['dropout']
    validation_split = 0.2
    n_steps = len(X_train) // n_batch
    n_val_steps = len(X_val) // n_batch
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
              #steps_per_epoch = n_steps-1,
              validation_data=(X_val, y_val),
              #validation_steps=n_val_steps-1,
              verbose=1,
              callbacks=[tensorboard_callback])

    return history, model, logdir


def run_image_model(X_train, X_val, y_train, y_val):

    from gbm_project.gen_params_cfg import gen_params, model_config
    
    gen_params['to_augment']=True

    batch_size = model_config['batch_size']
    n_epochs = model_config['n_epochs']
    learning_rate = model_config['learning_rate']
    dropout = model_config['dropout']
    batch_momentum = model_config['batch_momentum']
    op_momentum = model_config['op_momentum']
    if gen_params['to_augment']:
        steps_per_epoch = len(X_train)*(len(gen_params['augment_types'])+1) // batch_size
    else:
        steps_per_epoch = len(X_train) // batch_size
    val_steps = len(X_val) // batch_size

    #gen_params = {'batch_size': config['n_batch'],
    #              'data_dir': '../../data/upenn_GBM/numpy_conversion_downsample/',
    #              'modality': 'FLAIR',
    #              'dim': (70,86,82),
    #              'n_channels': 3,
    #              'n_classes': 2,
    #              'seed': 42,
    #              'shuffle': False}

    #train_generator = data_generator(X_train, y_train, **gen_params) 
    #val_generator = data_generator(X_val, y_val, **gen_params) 
    gen_params['to_augment']=True
    train_generator = DataGenerator(X_train, y_train, **gen_params)

    gen_params['to_augment']=False
    val_generator = DataGenerator(X_val, y_val, **gen_params) 

    Inputs = tf.keras.Input(shape=(*gen_params['dim'], gen_params['n_channels']), batch_size=batch_size)
    #Inputs = tf.keras.Input(shape=(70,86,82,3), batch_size=batch_size)

    #model = train_model_sequential(Inputs, dropout_rate=dropout)
    model = train_model_resnet(Inputs, dropout_rate=dropout, batchmomentum=batch_momentum)
    #model = train_model_resnet34(Inputs, dropout_rate=dropout, batchmomentum=batch_momentum)

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
    #model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=op_momentum),
    #model.compile(optimizer=tf.keras.optimizers.Adamax(learning_rate=learning_rate),
    #model.compile(optimizer=tf.keras.optimizers.Nadam(learning_rate=learning_rate),
    #model.compile(optimizer=tf.keras.optimizers.Ftrl(learning_rate=learning_rate),
                  loss='binary_crossentropy',
                  metrics=['accuracy', 'AUC'])
     
    print(model.summary())
    logdir = os.path.join('logs', datetime.now().strftime("%Y%m%d-%H%M%S"))

    tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)
    early_stopping_monitor = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(os.path.join(logdir, 'model_{epoch:03d}_{accuracy:0.2f}_{val_accuracy:0.2f}'), 
                                                          save_best_only=True, monitor='val_accuracy', verbose=0)

    #y_train_labels = y_train.iloc[:, 0].replace([1, 0], ['Methylated', 'Unmethylated'])

    y_train_labels = y_train.iloc[:, 0]
    class_weight = compute_class_weight('balanced', classes = np.unique(y_train_labels), y=y_train_labels)
    class_weight = {l : w for l,w in zip(np.unique(y_train_labels), class_weight)}

    history = model.fit(train_generator,
              epochs=n_epochs,
              validation_data=val_generator,
              steps_per_epoch = steps_per_epoch,
              validation_steps = val_steps,
              class_weight=class_weight,
              verbose=1,
              callbacks=[tensorboard_callback, model_checkpoint])
              #callbacks=[model_checkpoint])

    return history, model, logdir, gen_params, model_config


if __name__ == '__main__':
    #%tensorboard --logdir='./logs'
    
    csv_dir = '../../data/upenn_GBM/csvs/radiomic_features_CaPTk/'
    image_dir = '../../data/upenn_GBM/images/NIfTI-files/'
    out_dir = '../../data/upenn_GBM/numpy_conversion'
    modality = 'FLAIR'

    auto_df, man_df, comb_df = retrieve_data(csv_dir, modality=modality)
    patients = pd.DataFrame(auto_df.iloc[:, 0])

    image_T1_scaling = 'image_scaling_T1.json'
    image_T1GD_scaling = 'image_scaling_T1GD.json'
    image_T2_scaling = 'image_scaling_T2.json'
    image_FLAIR_scaling = 'image_scaling_FLAIR.json'
    
    X_train, X_test, X_val, y_train, y_test, y_val = split_image(patients)
    
    config_file = 'config_initial_20230118.json'
    with open(config_file, 'r') as f:
        config = json.load(f)
    
    n_batch = config['n_batch']
    n_epochs = config['n_epochs']
    learning_rate = config['learning_rate']
    dropout = config['dropout']
    gen_params = {'batch_size': config['n_batch'],
                  'data_dir': '../../data/upenn_GBM/numpy_conversion/',
                  'modality': 'T2',
                  'dim': (155,240,240),
                  'n_channels': 1,
                  'n_classes': 1,
                  'seed': 42,
                  'shuffle': False}
    
    history, model, logdir = run_image_model(config_file, image_FLAIR_scaling, X_train, X_val, y_train, y_val)
  
    #train_generator = data_generator(X_train, y_train, **gen_params) 
    #test_generator = data_generator(X_test, y_test, **gen_params) 

    #results_test = model.evaluate(test_generator)
    #results_train = model.evaluate(train_generator)
    #model.save(logdir)
    #print('evaluate on test data')
    #print('test loss, test acc, test auc:', results_test)
    #print('evaluate on train data')
    #print('train loss, train acc, train auc:', results_train)
    #y_pred = model.predict(X_test_adj, batch_size=n_batch)
    #y_pred_train = model.predict(X_train_adj, batch_size=n_batch)
    #display(y_pred)
    #display(y_pred_train)
