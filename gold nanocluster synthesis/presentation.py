# -*- coding: utf-8 -*-
"""
Created on Tue Nov  6 19:36:08 2018

@author: Leo
"""

from k_fold_cv import *
from keras.layers.advanced_activations import LeakyReLU

class LeakyReLU(LeakyReLU):
    def __init__(self, **kwargs):
        self.__name__ = "leakyrulu"
        super(LeakyReLU, self).__init__( **kwargs)

skf_sheet = ''

hparams = create_SNN_hparams(epochs=26, batch_size=32, pair_size=32, hidden_layers=[5], feature_vector_dim=49,
                             dropout=0.5, singlenet_l1=0.0, singlenet_l2=0.0, verbose=0,
                             fp_length=30, fp_number=3, conv_width=32, conv_number=2, conv_activation=LeakyReLU(),
                             conv_l1=0.001, conv_l2=0.001)

run_skf(model_mode='SNN_smiles', cv_mode='skf', hparams=hparams,
        loader_file='gold_data_loader.xlsm',
        skf_file='jiali_gold_skf.xlsx', skf_sheet=skf_sheet, k_folds=2, k_shuffle=True,
        save_model_name='Jiali_example', save_model=True)