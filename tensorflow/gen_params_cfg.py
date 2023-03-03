#!/usr/bin/env python

model_config = { 
	            "batch_size": 12,
	            "n_epochs": 40,
                "learning_rate": 1e-5,
                #"learning_rate": 0.01,
	            "dropout": 0.3,
                "batch_momentum": 0.99,
                "op_momentum": 0.8,
                "dilation": 1,
                "l2_reg": 1e-6
               }
gen_params = {
              'batch_size': model_config['batch_size'],
              #'data_dir': '../../data/upenn_GBM/numpy_conversion_window_predown/',
              #'data_dir': '../../data/upenn_GBM/numpy_conversion_downsample_pad/',
              'data_dir': '../../data/upenn_GBM/numpy_conversion_downsample_structure/',
              #'data_dir': '../../data/upenn_GBM/numpy_conversion_window/',
              'csv_dir': '../../data/upenn_GBM/csvs/radiomic_features_CaPTk/',
              'modality': 'FLAIR',
              #'dim': (140, 172, 164),
              #'dim': (70,86,82),
              #'dim': (70,86,86),
              'dim': (14,5,86,86),
              #'dim': (64,80,60),
              'n_channels': 3,
              'n_classes': 1,
              'seed': 42,
              'shuffle': True,
              'to_augment': True,
              'to_encode': False,
              'to_sectionate': True,
              'augment_types': ('flip', 'rotate', 'deform', 'noise')
              #'augment_types': ('flip', 'rotate', 'deform', 'noise', 'rotate2', 'rotate3', 'rotate4', 'rotate5')
             }

