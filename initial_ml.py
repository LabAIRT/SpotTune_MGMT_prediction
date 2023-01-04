#/usr/bin/env python

import os, json
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict

import tensorflow as tf
print("TensorFlow version:", tf.__version__)
from tensorflow.keras.layers import Dense, Convolution1D, LSTM, Flatten
from tensorflow.keras.layers import BatchNormalization, Dropout, Concatenate
from tensorflow.keras.models import Model
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from keras.metrics import AUC


def retrieve_data(csv_dir):
    # feature csv locations, genomic info is stored in the clinical info csv
    clinical_info = pd.read_csv(os.path.join(csv_dir, '../UPENN-GBM_clinical_info_v1.0.csv'))
    
    # maybe useful in the future, pulls all modalities and stores them into a dictionary of DataFrames 
    features_csvs = [os.path.join(csv_dir, f) for f in os.listdir(csv_dir) if f.endswith('.csv')]
    features_dfs = OrderedDict({os.path.split(f)[-1].strip('.csv'): pd.read_csv(f) for f in features_csvs})
    
    # Set the index to the Patient ID for ease of use and comparison
    clinical_info.set_index('ID', inplace=True)
    for f in features_dfs:
        features_dfs[f].set_index('SubjectID', inplace=True)
    
            # quick (and dirty) way of pulling specific modality dfs and storing them separately
    search_key_ed = '_automaticsegm_T1_ED'
    search_key_et = '_automaticsegm_T1_ET'
    search_key_nc = '_automaticsegm_T1_NC'
    t1_segm_ed = [df for key, df in features_dfs.items() if search_key_ed in key][0]
    t1_segm_et = [df for key, df in features_dfs.items() if search_key_et in key][0]
    t1_segm_nc = [df for key, df in features_dfs.items() if search_key_nc in key][0]
    
    ##################################################################################
    ######## rearrange some features that should be sequential #######################
    ##################################################################################
    match_str = 'Histogram_Bins-16_Bins-16_Bin-'
    match_hist_str = 'Histogram_Bins-16_Bins-16_'
    
    # retrieve list of column names that need to be sequential
    columns_to_sort_ed = [col for col in t1_segm_ed.columns if match_str in col]
    columns_to_sort_et = [col for col in t1_segm_et.columns if match_str in col]
    columns_to_sort_nc = [col for col in t1_segm_nc.columns if match_str in col]
    
    # sort those column names based on 2 criteria
    #   - Freq vs Prob
    #   - Bin number
    col_sort_ed = sorted(columns_to_sort_ed, key=lambda x: (x.split('_')[-1], int(x.split('_')[-2].split('-')[-1])))
    col_sort_et = sorted(columns_to_sort_et, key=lambda x: (x.split('_')[-1], int(x.split('_')[-2].split('-')[-1])))
    col_sort_nc = sorted(columns_to_sort_nc, key=lambda x: (x.split('_')[-1], int(x.split('_')[-2].split('-')[-1])))
    
    # separate out the columns into 3 groups: histogram bin values, other histogram features, everything else
    t1_segm_ed_diff = t1_segm_ed[t1_segm_ed.columns.difference(col_sort_ed)]
    col_rem_hist_ed = [col for col in t1_segm_ed_diff.columns if match_hist_str in col]
    t1_segm_ed_diff_part2 = t1_segm_ed_diff[t1_segm_ed_diff.columns.difference(col_rem_hist_ed)]
    t1_segm_ed_remhist = t1_segm_ed_diff[col_rem_hist_ed]
    t1_segm_ed_sorted = t1_segm_ed[col_sort_ed]
    # join the 3 groups back together, this time in the desired sequential order
    t1_segm_ed = t1_segm_ed_diff_part2.join(t1_segm_ed_sorted, how='inner').join(t1_segm_ed_remhist, how='inner')
    
    # repeat for other tumor segments
    t1_segm_et_diff = t1_segm_et[t1_segm_et.columns.difference(col_sort_et)]
    col_rem_hist_et = [col for col in t1_segm_et_diff.columns if match_hist_str in col]
    t1_segm_et_diff_part2 = t1_segm_et_diff[t1_segm_et_diff.columns.difference(col_rem_hist_et)]
    t1_segm_et_remhist = t1_segm_et_diff[col_rem_hist_et]
    t1_segm_et_sorted = t1_segm_et[col_sort_et]
    t1_segm_et = t1_segm_et_diff_part2.join(t1_segm_et_sorted, how='inner').join(t1_segm_et_remhist, how='inner')
    
    t1_segm_nc_diff = t1_segm_nc[t1_segm_nc.columns.difference(col_sort_nc)]
    col_rem_hist_nc = [col for col in t1_segm_nc_diff.columns if match_hist_str in col]
    t1_segm_nc_diff_part2 = t1_segm_nc_diff[t1_segm_nc_diff.columns.difference(col_rem_hist_nc)]
    t1_segm_nc_remhist = t1_segm_nc_diff[col_rem_hist_nc]
    t1_segm_nc_sorted = t1_segm_nc[col_sort_nc]
    t1_segm_nc = t1_segm_nc_diff_part2.join(t1_segm_nc_sorted, how='inner').join(t1_segm_nc_remhist, how='inner')
    #################################################################################
    #################################################################################
        
    # pulling the manually refined segmentation data
    search_key_ed = '_segm_T1_ED'
    search_key_et = '_segm_T1_ET'
    search_key_nc = '_segm_T1_NC'
    t1_man_segm_ed = [df for key, df in features_dfs.items() if search_key_ed in key][0]
    t1_man_segm_et = [df for key, df in features_dfs.items() if search_key_et in key][0]
    t1_man_segm_nc = [df for key, df in features_dfs.items() if search_key_nc in key][0]
    
    ##################################################################################
    ######## rearrange some features that should be sequential for manually segmented data #######################
    ##################################################################################
    t1_man_segm_ed_diff = t1_man_segm_ed[t1_man_segm_ed.columns.difference(col_sort_ed)]
    t1_man_segm_ed_diff_part2 = t1_man_segm_ed_diff[t1_man_segm_ed_diff.columns.difference(col_rem_hist_ed)]
    t1_man_segm_ed_remhist = t1_man_segm_ed_diff[col_rem_hist_ed]
    t1_man_segm_ed_sorted = t1_man_segm_ed[col_sort_ed]
    t1_man_segm_ed = t1_man_segm_ed_diff_part2.join(t1_man_segm_ed_sorted, how='inner').join(t1_man_segm_ed_remhist, how='inner')
    
    t1_man_segm_et_diff = t1_man_segm_et[t1_man_segm_et.columns.difference(col_sort_et)]
    t1_man_segm_et_diff_part2 = t1_man_segm_et_diff[t1_man_segm_et_diff.columns.difference(col_rem_hist_et)]
    t1_man_segm_et_remhist = t1_man_segm_et_diff[col_rem_hist_et]
    t1_man_segm_et_sorted = t1_man_segm_et[col_sort_et]
    t1_man_segm_et = t1_man_segm_et_diff_part2.join(t1_man_segm_et_sorted, how='inner').join(t1_man_segm_et_remhist, how='inner')
    
    t1_man_segm_nc_diff = t1_man_segm_nc[t1_man_segm_nc.columns.difference(col_sort_nc)]
    t1_man_segm_nc_diff_part2 = t1_man_segm_nc_diff[t1_man_segm_nc_diff.columns.difference(col_rem_hist_nc)]
    t1_man_segm_nc_remhist = t1_man_segm_nc_diff[col_rem_hist_nc]
    t1_man_segm_nc_sorted = t1_man_segm_nc[col_sort_nc]
    t1_man_segm_nc = t1_man_segm_nc_diff_part2.join(t1_man_segm_nc_sorted, how='inner').join(t1_man_segm_nc_remhist, how='inner')
    #################################################################################
    #################################################################################
        
    # pull out the MGMT labels from clinical info, convert to dummy values (0s and 1s) for classification, drop Not Available
    genomics = clinical_info['MGMT']
    truthy_dummies = pd.get_dummies(genomics)
    mgmt_class = truthy_dummies[['Methylated', 'Unmethylated']][np.logical_and(truthy_dummies['Not Available'] != 1, truthy_dummies['Indeterminate'] != 1)]
    
    # match and join classifiers with features for the complete dataframe 
    t1_mgmt_df = mgmt_class.join(t1_segm_ed, how='inner').join(t1_segm_et, how='inner').join(t1_segm_nc, how='inner')
    t1_man_mgmt_df = mgmt_class.join(t1_man_segm_ed, how='inner').join(t1_man_segm_et, how='inner').join(t1_man_segm_nc, how='inner')
    
    
    man_index = t1_man_mgmt_df.index.values
    # make a deep copy so as not to affect the original df
    t1_comb_mgmt_df = t1_mgmt_df.copy(deep=True)
    
    # need to convert the values to be changed to NaNs (essentially empty the values) so that combine_first can be used
    for ind in man_index:
        t1_comb_mgmt_df.loc[ind] = np.nan
        
    # fills all values of NaN in the first df with the values from the second
    t1_comb_mgmt_df = t1_comb_mgmt_df.combine_first(t1_man_mgmt_df)
    
    t1_mgmt_df[t1_mgmt_df.isnull().any(axis=1)].index.tolist()
    
    # dropping rows with nan values
    to_drop = t1_mgmt_df[t1_mgmt_df.isnull().any(axis=1)].index.tolist()
    t1_mgmt_df.drop(labels=to_drop,
                    axis=0,
                    inplace=True)

    # feature selection, dropping unneeded features
    feature_to_remove = ['_OrientedBoundingBoxSize',
                         '_PerimeterOnBorder',
                         '_PixelsOnBorder',
                         'Bins-16_Maximum',
                         'Bins-16_Minimum',
                         'Bins-16_Range']

    feature_mask = t1_mgmt_df.columns.str.contains('|'.join(feature_to_remove))
    feature_column_names = t1_mgmt_df.loc[:, feature_mask].columns.tolist()
    # Don't use difference, it shuffles the columns
    t1_mgmt_df.drop(feature_column_names, axis=1, inplace=True)
    t1_man_mgmt_df.drop(feature_column_names, axis=1, inplace=True)
    t1_comb_mgmt_df.drop(feature_column_names, axis=1, inplace=True)

    return t1_mgmt_df, t1_man_mgmt_df, t1_comb_mgmt_df


#######################################################################################
# split df into X and y set
def scale_and_split(df, n_cat=2):
    """
    splits and scales input dataframe# and outputs as ndarray, assumes binary categories in the first two columns of the dataframe
    """
    # separate out the inputs and lab#els 
    y = df.iloc[:, :n_cat]
    X = df.iloc[:, n_cat:]

    X_column_names = X.columns.tolist()
    X_pat_ids = X.index.tolist()
    y_column_names = y.columns.tolist()
    y_pat_ids = y.index.tolist()
    # make a mask separating out tumor sections
    ed_mask = X.columns.str.contains('_ED_', regex=False)
    et_mask = X.columns.str.contains('_ET_', regex=False)
    nc_mask = X.columns.str.contains('_NC_', regex=False)
    
    # scale X data to 0 mean and unit variance (standard scaling)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # need to nest the tumor section features into three groups per patient
    X_scaled_ed = X_scaled[:, ed_mask]
    X_scaled_et = X_scaled[:, et_mask]
    X_scaled_nc = X_scaled[:, nc_mask]

    data_comb = np.array([[ed, et, nc] for ed, et, nc in zip(X_scaled_ed, X_scaled_et, X_scaled_nc)])
    X_scaled_comb = data_comb
    
    # add extra empty dimension so the conv1D wont yell at us
    X_scaled_comb_extradim = np.expand_dims(X_scaled_comb, axis=3)
    y = y.to_numpy()
    
    # Separate into train and test datasets.
    # train_test_split automatically shuffles and splits the data following predefined sizes can revisit if shuffling is not a good idea
    X_train, X_test, y_train, y_test = train_test_split(X_scaled_comb_extradim, y, test_size=0.3, random_state=42, stratify=y)
    
    #X_train = X_train[:-2] # remove some patients so it is divisble by 11
    #y_train = y_train[:-2]

    return X_train, X_test, y_train, y_test, X_scaled, y



def get_pc(df, components=0.9):
    pca = PCA(n_components=components, random_state=42)
    df_transform = pca.fit_transform(df)
    return df_transform, pca


def drop_corr(df, corr_cut = 0.9)
    corr_df = df.corr().abs()
    upper_mask = np.triu(np.ones_like(corr_df, dtype=bool))
    upper_corr_df = corr_df.mask(upper_mask)
    col_to_drop = [c for c in upper_corr_df.columns if np.any(upper_corr_df[c] > corr_cut)]

    df_reduced = df.drop(col_to_drop, axis=1)
    return df_reduced, col_to_drop


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
        et = Convolution1D(64, 1, kernel_initializer='lecun_uniform', activation='relu', name='et_conv0')(et)
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
        nc = Convolution1D(64, 1, kernel_initializer='lecun_uniform', activation='relu', name='nc_conv0')(nc)
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
        nc = LSTM(150, go_backwards=True, implementation=2, name='nc_lstm')(nc)
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
    pred = Dense(2, activation='sigmoid', kernel_initializer='lecun_uniform', name='ID_pred')(x_comb)

    model = Model(Inputs, pred)
    return model


def run_model(config_file, X_train, y_train):
    with open(config_file, 'r') as f:
        config = json.load(f)

    n_batch = config['n_batch']
    n_epochs = config['n_epochs']
    learning_rate = config['learning_rate']
    dropout = config['dropout']

    # set up the keras Input object, shape corresponds to the 3x144 array per patient in X_scaled
    Inputs = tf.keras.Input(shape=(3,144,1), batch_size=n_batch)

    model = train_model(Inputs, dropout_rate=dropout, rec_active=False, dense_active=False)

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

    return history, model


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
    
    #y_pred = model.predict(X_test_adj, batch_size=n_batch)
    #y_pred_train = model.predict(X_train, batch_size=n_batch)
    #auc = AUC()
    #auc.update_state(y_test_adj, y_pred)
    #print(auc.result())
    #auc_train = AUC()
    #auc_train.update_state(y_train, y_pred_train)
    #print(auc_train.result())
    #
    #results_test = model.evaluate(X_test_adj, y_test_adj, batch_size=n_batch)
    #results_train = model.evaluate(X_train, y_train, batch_size=n_batch)
    #model.save(logdir)
    #print('evaluate on test data')
    #print('test loss, test acc, test auc:', results_test)
    #print('evaluate on train data')
    #print('train loss, train acc, train auc:', results_train)


