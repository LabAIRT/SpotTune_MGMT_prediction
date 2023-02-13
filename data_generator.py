#!/usr/bin/env python
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from skimage.util import random_noise
from skimage.transform import rotate
from sklearn.preprocessing import StandardScaler
import elasticdeform
from gbm_project.data_prep import retrieve_data 

class DataGenerator(tf.keras.utils.Sequence):
    """
    Generate data for a Keras model
    """
    def __init__(self, data_indices, labels, csv_dir='../../data/upenn_GBM/csvs/radiomic_features_CaPTk/', data_dir='../../data/upenn_GBM/images/NIfTI-files/', modality='T2', batch_size=11, dim=(155, 240, 240), n_channels=1, n_classes=2, shuffle=True, to_augment=False, augment_types=('noise', 'flip', 'rotate', 'deform'), seed=42, to_encode=False):
        """
        Initialization
        """
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.data_indices = data_indices
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle=shuffle
        self.rng_shuffle = np.random.default_rng(seed)
        self.rng_noise = np.random.default_rng(seed)
        self.rng_rotate = np.random.default_rng(seed)
        self.data_dir = data_dir
        self.csv_dir = csv_dir
        self.modality = modality
        self.to_augment = to_augment
        self.augment_types = augment_types
        self.to_encode = to_encode
        self.radiomics = None

        if self.to_encode:
            temp_radiomics = retrieve_data(self.csv_dir, self.modality)[0]
            self.radiomics = temp_radiomics.loc[self.data_indices, 2:]
            scaler = StandardScaler()

            temp_index = self.radiomics.index.tolist()
            temp_column = self.radiomics.columns.tolist()
            temp_scaled = scaler.fit_transform(self.radiomics)

            self.radiomics = pd.DataFrame(temp_scaled, columns=temp_column, index=temp_index)


        if self.to_augment:
            augment_idx = {}
            for aug in self.augment_types:
                augment_idx[aug] = self.labels.copy(deep=True)
                augment_idx[aug].index = augment_idx[aug].index+'_'+aug

                self.data_indices = self.data_indices.append(augment_idx[aug].index)

            self.labels = pd.concat([self.labels, *augment_idx.values()])

        self.on_epoch_end()


    def __len__(self):
        """
        Number of batches per epoch
        """
        return int(np.floor(len(self.data_indices) / self.batch_size))


    def __getitem__(self, index):
        """
        Generate one batch of data
        """
        indices = self.indices[index*self.batch_size:(index+1)*self.batch_size]

        data_indices_temp = [self.data_indices[k] for k in indices]

        X, y = self.__data_generation(data_indices_temp)

        return X, y


    def on_epoch_end(self):
        """
        Update indices after each epoch
        """
        self.indices = np.arange(len(self.data_indices))
        if self.shuffle:
            self.rng_shuffle.shuffle(self.indices)


    def __data_generation(self, data_indices_temp):
        """
        Generator data containing batch_size samples
        """
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size, self.n_classes), dtype=int)

        for i, idx in enumerate(data_indices_temp):

            if self.to_augment:
                if len(idx.split('_')) == 3:
                    in_arr = np.load(os.path.join(self.data_dir,'_'.join(idx.split('_')[:2])+'_'+self.modality+ '.npy'))
                else:
                    in_arr = np.load(os.path.join(self.data_dir,idx+'_'+self.modality+ '.npy'))
            else:
                in_arr = np.load(os.path.join(self.data_dir,idx+'_'+self.modality+ '.npy'))

            # take the first three images such that they are arranged into a single image with 3 channels.`
            in_arr = np.transpose(in_arr[0:3], axes=[1,2,3,0])

            aug = idx.split('_')[-1]
            if aug in self.augment_types:
                if 'flip' in aug:
                    in_arr = self.apply_flip(in_arr)
                if 'rotate' in aug:
                    in_arr = self.apply_rotation(in_arr)
                if 'noise' in aug:
                    in_arr = self.apply_noise(in_arr)
                if 'deform' in aug:
                    in_arr = self.apply_deformation(in_arr)

            X[i,] = in_arr
            #X[i,] = np.transpose(in_arr, axes=[1,2,3,0])
            y[i] = self.labels.loc[idx]

        #return (X, tf.keras.utils.to_categorical(y, num_classes=self.n_classes))
        return (X, y)


    def apply_noise(self, arr):
        return random_noise(arr, mode='gaussian', seed=self.rng_noise)


    def apply_rotation(self, arr):
        angle = self.rng_rotate.integers(-180, high=180)
        arr[:,:,:,0] = rotate(arr[:,:,:,0], angle, preserve_range=True)
        arr[:,:,:,1] = rotate(arr[:,:,:,1], angle, preserve_range=True)
        arr[:,:,:,2] = rotate(arr[:,:,:,2], angle, preserve_range=True)
        return arr


    def apply_deformation(self, arr):
        return elasticdeform.deform_random_grid(arr, sigma=5, order=0, axis=(0,1,2))


    def apply_flip(self, arr):
        return np.flip(arr, axis=(0,1,2))


    def encode(self, arr, idx):
        arr.append(np.zeros((self.dim[1], self.dim[2], self.n_channels)), axis=0)

        ed_arr = self.radiomics.loc[idx].iloc[:, self.radiomics.columns.str.contains('_ED_', regex=False)] 
        et_arr = self.radiomics.loc[idx].iloc[:, self.radiomics.columns.str.contains('_ET_', regex=False)] 
        nc_arr = self.radiomics.loc[idx].iloc[:, self.radiomics.columns.str.contains('_NC_', regex=False)] 

        ed_arr = ed_arr.reshape((15,9))
        et_arr = et_arr.reshape((15,9))
        nc_arr = nc_arr.reshape((15,9))

        arr[-1, (43-7):(41+8), (41-4):(41+5), 0] = ed_arr
        arr[-1, (43-7):(41+8), (41-4):(41+5), 1] = et_arr
        arr[-1, (43-7):(41+8), (41-4):(41+5), 2] = nc_arr

        return arr



def data_generator(data_indices, labels, data_dir='../../data/upenn_GBM/images/NIfTI-files/', modality='T2', batch_size=8, dim=(155, 240, 240), n_channels=1, n_classes=2, shuffle=True, seed=42):
    """
    generator function that doesn't rely on keras Sequence() for multi-input training
    """

    rng = np.random.default_rng(seed)
    data_indices = np.array(data_indices)
    batch_per_epoch = int(np.floor(len(data_indices)) / batch_size)

    while True:
        idx_step = 0
        n_batch = 0
        while n_batch < batch_per_epoch:
            X = np.empty((batch_size, *dim))
            y = np.empty((batch_size), dtype=int)

            data_indices_temp = data_indices[idx_step:idx_step+batch_size]

            for i, idx in enumerate(data_indices_temp):
                in_arr = np.load(os.path.join(data_dir,idx+'_'+modality+ '.npy'))
                # take the first three images such that they are arranged into a single image with 3 channels.`
                X[i,] = np.transpose(in_arr[0:3], axes=[1,2,3,0])
                y[i] = int(labels.loc[idx])

            n_batch += 1
            yield (X, tf.keras.utils.to_categorical(y, num_classes=n_classes))

        if shuffle:
            rng.shuffle(data_indices)











