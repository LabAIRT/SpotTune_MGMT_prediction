#!/usr/bin/env python

model_config = { 
	            "batch_size": 12,
	            "n_epochs": 30,
                "learning_rate": 1e-4,
                #"learning_rate": 0.01,
	            "dropout": 0.3,
                "dilation": 1,
                "l2_reg": 1e-6,
                "dim": (70, 86, 86),
                "lr_step_size": 15,
                "lr_gamma": 0.1,
                "no_freeze": ['conv_seg', 'layer4']
               }


gen_params = {
              'data_dir': '../../data/upenn_GBM/numpy_conversion_mod_channels/',
              #'data_dir': '../../data/upenn_GBM/numpy_conversion_downsample_structure/',
              'csv_dir': '../../data/upenn_GBM/csvs/radiomic_features_CaPTk/',
              #'modality': ['FLAIR','T2','T1'],
              'modality': ['mods'],
              #'dim': (140, 172, 164),
              #'dim': (70,86,82),
              'dim': (70,86,86),
              #'dim': (14,5,86,86),
              #'dim': (64,80,60),
              'n_channels': 3,
              'seed': 42,
              'to_augment': True,
              'to_encode': False,
              'to_sectionate': False,
              'augment_types': ('flip', 'rotate', 'deform', 'noise')
              #'augment_types': ('flip', 'rotate', 'deform', 'noise', 'rotate2', 'rotate3', 'rotate4', 'rotate5')
             }
