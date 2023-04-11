#!/usr/bin/env python

model_config = { 
	            "batch_size": 12,
	            "n_epochs": 30,
                "learning_rate": 1e-5,
                #"learning_rate": 0.01,
	            "dropout": 0.3,
                "dilation": 1,
                "l2_reg": 1e-8,
                "dim": (70, 86, 86),
                "agent_learning_rate": 0.0001,
                "agent_l2_reg": 1e-8,
                "no_freeze": ['conv_seg', 'layer4'],
                "spottune": False,
                "lr_sched": False,
                "lr_factor": 0.1,
                "lr_steps": [60, 80],
                "lr_patience": 5
               }


gen_params = {
              'data_dir': '../../data/upenn_GBM/numpy_conversion_man_DSC_channels/',
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
              'seed': 42,
              'to_augment': True,
              'to_encode': False,
              'to_slice': False,
              'n_slices': 10,
              'augment_types': ('flip', 'rotate', 'deform')
              #'augment_types': ('flip', 'rotate', 'noise', 'deform')
              #'augment_types': ('flip', 'rotate', 'noise')
              #'augment_types': ('flip', 'rotate', 'noise', 'deform', 'flip+rotate', 'flip+noise', 'flip+deform', 'rotate+noise', 'rotate+deform', 'noise+deform')
              #'augment_types': ('flip', 'rotate', 'deform', 'noise')
             }
