#!/usr/bin/env python

model_config = { 
	            "batch_size": 12,
	            "n_epochs": 50,
                "learning_rate": 0.0005,
	            "dropout": 0.5,
                "batch_momentum": 0.9,
                "op_momentum": 0.8,
               }
gen_params = {
              'batch_size': model_config['batch_size'],
              #'data_dir': '../../data/upenn_GBM/numpy_conversion_window_predown/',
              'data_dir': '../../data/upenn_GBM/numpy_conversion_downsample_scalefix/',
              #'data_dir': '../../data/upenn_GBM/numpy_conversion_window/',
              'modality': 'FLAIR',
              #'dim': (140, 172, 164),
              'dim': (70,86,82),
              #'dim': (64,80,60),
              'n_channels': 3,
              'n_classes': 1,
              'seed': 42,
              'shuffle': True,
              'to_augment': True,
              #'augment_types': ('noise', 'flip', 'rotate', 'noise2', 'noise3')
              'augment_types': ('noise', 'flip', 'rotate', 'deform')
             }

