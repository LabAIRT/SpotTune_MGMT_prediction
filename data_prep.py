#!/usr/bin/env/python

import os, json
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


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
    mgmt_class = truthy_dummies[['Methylated']][np.logical_and(truthy_dummies['Not Available'] != 1, truthy_dummies['Indeterminate'] != 1)]
    
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
def scale_and_split(df, do_pca=False, do_corr=False, n_cat=1):
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

    # apply pca to reduce dimensionality
    # need to figure out how to get this to work with a jagged array
    if do_pca:
        X_ed_reduced, comp_ed = get_pc(X_scaled_ed)
        n_components = comp_ed.n_components_
        X_et_reduced, comp_et = get_pc(X_scaled_et, n_components)
        X_nc_reduced, comp_nc = get_pc(X_scaled_nc, n_components)
    elif do_corr:
        X_ed_reduced, comp_ed = drop_corr(X_scaled_ed)
        X_et_reduced, comp_et = drop_corr(X_scaled_et)
        X_nc_reduced, comp_nc = drop_corr(X_scaled_nc)
    else:
        X_ed_reduced = X_scaled_ed
        X_et_reduced = X_scaled_et
        X_nc_reduced = X_scaled_nc

    data_comb = np.array([[ed, et, nc] for ed, et, nc in zip(X_ed_reduced, X_et_reduced, X_nc_reduced)])
    X_scaled_comb = data_comb
    

    # add extra empty dimension so the conv1D wont yell at us
    X_scaled_comb_extradim = np.expand_dims(X_scaled_comb, axis=3)
    y = y.to_numpy()
    
    # Separate into train and test datasets.
    # train_test_split automatically shuffles and splits the data following predefined sizes can revisit if shuffling is not a good idea
    X_train, X_test, y_train, y_test = train_test_split(X_scaled_comb_extradim, y, test_size=0.3, random_state=42, stratify=y)
    
    #X_train = X_train[:-2] # remove some patients so it is divisble by 11
    #y_train = y_train[:-2]

    return X_train, X_test, y_train, y_test, X_scaled, y, n_components



def scale_and_split_comb(df, do_pca=False, do_corr=False, n_cat=1):
    """
    splits and scales input dataframe# and outputs as ndarray, assumes binary categories in the first column
    """
    # separate out the inputs and lab#els 
    y = df.iloc[:, :n_cat]
    X = df.iloc[:, n_cat:]

    # scale X data to 0 mean and unit variance (standard scaling)
    scaler = StandardScaler()

    # apply pca to reduce dimensionality
    # need to figure out how to get this to work with a jagged array
    if do_pca:
        X_scaled = scaler.fit_transform(X)
        X_reduced, comp = get_pc(X_scaled)
        n_components = comp.n_components_
    elif do_corr:
        X_reduced, comp = drop_corr(X)
        n_components = comp
        X_scaled = scaler.fit_transform(X_reduced)
        X_reduced = X_scaled
    else:
        X_reduced = X_scaled
        n_components = len(X_reduced[0])

    # add extra empty dimension so the conv1D wont yell at us
    X_scaled_extradim = np.expand_dims(X_reduced, axis=2)
    y = y.to_numpy()
    
    # Separate into train and test datasets.
    # train_test_split automatically shuffles and splits the data following predefined sizes can revisit if shuffling is not a good idea
    X_train, X_test, y_train, y_test = train_test_split(X_scaled_extradim, y, test_size=0.3, random_state=42, stratify=y)
    
    #X_train = X_train[:-2] # remove some patients so it is divisble by 11
    #y_train = y_train[:-2]

    return X_train, X_test, y_train, y_test, X_scaled, y, n_components


def get_pc(df, components=0.9):
    pca = PCA(n_components=components, random_state=42)
    df_transform = pca.fit_transform(df)
    return df_transform, pca


def drop_corr(df, corr_cut = 0.5):
    corr_df = df.corr().abs()
    upper_mask = np.triu(np.ones_like(corr_df, dtype=bool))
    upper_corr_df = corr_df.mask(upper_mask)
    col_to_drop = [c for c in upper_corr_df.columns if np.any(upper_corr_df[c] > corr_cut)]

    df_reduced = df.drop(col_to_drop, axis=1)
    return df_reduced, len(df_reduced.columns)


