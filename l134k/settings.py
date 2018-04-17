#!/usr/bin/env python
# -*- coding:utf-8 -*-

# Tensor settings
tensor_settings = {
    'padding_final_size': 30,
    'add_extra_atom_attribute': True,
    'add_extra_bond_attribute': True,
}

# Model settings
model_settings = {
    'embedding_size': 512,
    'depth': 3,
    'scale_output': 0.05,
    'padding': True,
    'mol_conv_inner_activation': 'tanh',
    'mol_conv_outer_activation': 'softmax',
    'hidden': 50,
    'hidden_activation': 'tanh',
    'output_activation': 'linear',
    'output_size': 1,
    'lr': 0.01,
    'optimizer': 'adam',
    'loss': 'mse',
}

# Training settings
train_settings = {
    'nb_epoch': 150,
    'batch_size': 1,
    'lr_func': "float({0} * np.exp(- epoch / {1}))".format(0.0007, 30.0),
    'patience': 10,
}
