#!/usr/bin/env python

model_config = { 
	            "batch_size": 12,
	            "n_epochs": 60,
                #"learning_rate": 0.01,
	            "dropout": 0.0,
	            "agent_dropout": 0.0,
                "dilation": 1,
                "l2_reg": 1e-8,
                "dim": (70, 86, 86),
                "learning_rate": 1e-5,
                "agent_learning_rate": 1e-4,
                "agent_l2_reg": 1e-8,
                "no_freeze": ['conv_seg'],
                "spottune": True,
                "lr_sched": True,
                "lr_factor": 0.1,
                "lr_steps": [30],
                "lr_patience": 5,
                "gumbel_temperature": 1e4,
                "temp_steps": [0],
                "temp_vals": [1e4],
                "seed_switch": "high", # low/mid/high randomness
                "seed_steps": [0, 50],
                "seed_vals": ["high", "med"],
                "log_dir": None
               }


gen_params = {
              'data_dir': '../../data/upenn_GBM/numpy_conversion_DSC_augmented_channels/',
              #'data_dir': '../../data/upenn_GBM/numpy_conversion_man_DTI_channels/',
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
              'n_classes': 1,
              'seed': 42,
              'to_augment': True,
              'make_augment': False,
              'to_encode': False,
              'to_slice': False,
              'to_3D_slice': False,
              'n_slices': 10,
              #'augment_types': ('flip', 'rotate', 'deform')
              'augment_types': ('flip', 'rotate', 'noise', 'deform')
              #'augment_types': ('flip', 'rotate', 'noise', 'deform', 'rotate2', 'rotate3', 'rotate4', 'rotate5')
              #'augment_types': ('flip', 'rotate', 'noise', 'deform', 'flip+rotate', 'flip+noise', 'flip+deform', 'rotate+noise', 'rotate+deform', 'noise+deform')
              #'augment_types': ('flip', 'rotate', 'deform', 'noise')
             }
