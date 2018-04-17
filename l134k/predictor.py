#!/usr/bin/env python
# -*- coding:utf-8 -*-

import logging
import os
import shutil

from rmgpy.cnn_framework.cnn_model import build_model, train_model, save_model
from rmgpy.cnn_framework.molecule_tensor import get_attribute_vector_size
from rmgpy.cnn_framework.data import split_test_from_train_and_val, split_inner_val_from_train_data


class Predictor(object):
    def __init__(self, out_dir):
        self.model = None
        self.out_dir = out_dir

    def build_model(self, tensor_settings, **model_settings):
        attribute_vector_size = get_attribute_vector_size(
            add_extra_atom_attribute=tensor_settings['add_extra_atom_attribute'],
            add_extra_bond_attribute=tensor_settings['add_extra_bond_attribute']
        )
        self.model = build_model(attribute_vector_size=attribute_vector_size, **model_settings)

    def load_weights(self, model_weights_path):
        logging.info('Loading model weights from {}'.format(model_weights_path))
        self.model.load_weights(model_weights_path)

    def full_train(self, x, y, names, test_split, save_names=False, **train_settings):
        logging.info('Splitting dataset with a test split of {}'.format(test_split))
        x_test, y_test, x_train, y_train, names_test, names_train = split_test_from_train_and_val(
            x, y, extra_data=names, shuffle_seed=7, testing_ratio=test_split
        )

        if save_names:
            names_test_path = os.path.join(self.out_dir, 'names_test.txt')
            names_train_path = os.path.join(self.out_dir, 'names_train.txt')
            with open(names_test_path, 'w') as f:
                for name in names_test:
                    f.write(name + '\n')
            with open(names_train_path, 'w') as f:
                for name in names_train:
                    f.write(name + '\n')

        logging.info('Splitting training data into early-stopping validation and remaining training sets')
        x_train, x_inner_val, y_train, y_inner_val = split_inner_val_from_train_data(x_train, y_train,
                                                                                     shuffle_seed=77,
                                                                                     training_ratio=0.9)

        logging.info('Training model...')
        logging.info('Training data: {} points'.format(len(x_train)))
        logging.info('Inner validation data: {} points'.format(len(x_inner_val)))
        logging.info('Test data: {} points'.format(len(x_test)))
        self.model, loss, inner_val_loss, mean_outer_val_loss, mean_test_loss = train_model(
            self.model, x_train, y_train, x_inner_val, y_inner_val, x_test, y_test,
            X_outer_val=None, y_outer_val=None, **train_settings
        )

        self.save_model(loss, inner_val_loss, mean_outer_val_loss, mean_test_loss)

    def save_model(self, loss, inner_val_loss, mean_outer_val_loss, mean_test_loss):
        logging.info('Saving model')
        model_path = os.path.join(self.out_dir, 'model')
        model_structure_path = model_path + '.json'
        model_weights_path = model_path + '.h5'
        if os.path.exists(model_structure_path):
            logging.info(
                'Backing up model structure (and removing old backup if present): {}'.format(model_structure_path))
            shutil.move(model_structure_path, model_path + '_backup.json')
        if os.path.exists(model_weights_path):
            logging.info('Backing up model weights (and removing old backup if present): {}'.format(model_path + '.h5'))
            shutil.move(model_weights_path, model_path + '_backup.h5')
        save_model(self.model, loss, inner_val_loss, mean_outer_val_loss, mean_test_loss, model_path)
