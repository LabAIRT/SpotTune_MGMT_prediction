#!/usr/bin/env python

model_config = { 
	            "batch_size": 12,
	            "n_epochs": 30,
                "learning_rate": 0.00001,
	            "dropout": 0.4,
                "batch_momentum": 0.99,
                "op_momentum": 0.8,
                "dilation": 1,
                "l2_reg": 0.00000001
               }
gen_params = {
              'batch_size': model_config['batch_size'],
              #'data_dir': '../../data/upenn_GBM/numpy_conversion_window_predown/',
              'data_dir': '../../data/upenn_GBM/numpy_conversion_downsample_pad/',
              #'data_dir': '../../data/upenn_GBM/numpy_conversion_window/',
              'csv_dir': '../../data/upenn_GBM/csvs/radiomic_features_CaPTk/',
              'modality': 'FLAIR',
              #'dim': (140, 172, 164),
              #'dim': (70,86,82),
              'dim': (70,86,86),
              #'dim': (64,80,60),
              'n_channels': 3,
              'n_classes': 1,
              'seed': 42,
              'shuffle': True,
              'to_augment': True,
              'to_encode': False,
              'augment_types': ('flip', 'rotate', 'deform', 'noise')
              #'augment_types': ('noise', 'flip', 'rotate', 'deform')
             }

